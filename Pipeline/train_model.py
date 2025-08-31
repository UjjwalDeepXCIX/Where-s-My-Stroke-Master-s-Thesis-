# 5_train_model.py — corrected

import os, time, glob, re, logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib, json
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.manifold import TSNE
from scipy.signal import welch, coherence
from scipy.stats import entropy, skew, kurtosis
from pywt import cwt

# ─── PATHS ──────────────────────────────────────────────────────────────────────
EPOCHS_DIR    = r"C:\UoM Sem-2\Research_work\src\epochs"
FEATURE_DIR   = r"C:\UoM Sem-2\Research_work\src\features"
PLOTS_DIR     = r"C:\UoM Sem-2\Research_work\src\plots\training"
LOGS_DIR      = r"C:\UoM Sem-2\Research_work\src\logs"
MODEL_DIR     = r"C:\UoM Sem-2\Research_work\src\models"
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(LOGS_DIR, exist_ok=True)

# ─── LOGGING ────────────────────────────────────────────────────────────────────
log_file = os.path.join(LOGS_DIR, f"train_{int(time.time())}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
use_metadata = True
META_TSV = r"C:\UoM Sem-2\Research_work\Datasets\Dataset\FIgshare\metadata\participants.tsv"

channels = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']
freq_bands = {'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 100.0)}
sfreq = 256.0
min_channels = 10

TRAIN_NORMAL  = os.path.join(EPOCHS_DIR, "train", "normal")
TRAIN_STROKE  = os.path.join(EPOCHS_DIR, "train", "stroke")
EVAL_NORMAL   = os.path.join(EPOCHS_DIR, "eval", "normal")
EVAL_STROKE   = os.path.join(EPOCHS_DIR, "eval", "stroke")

# ─── HELPERS: metadata weights ─────────────────────────────────────────────────
def get_feature_weights(participant_id):
    if not use_metadata or participant_id is None:
        return 1.0
    meta = pd.read_csv(META_TSV, sep="\t", index_col=0)
    if participant_id not in meta.index:
        return 1.0
    nihss = meta.loc[participant_id, 'NIHSS']
    sloc  = str(meta.loc[participant_id, 'StrokeLocation']).lower()
    w = 1.0 + (nihss / 11.0)
    if ('pons' in sloc) or ('basal ganglia' in sloc):
        w *= 1.2
    return float(w)

# ─── FEATURE BUILDERS (same shapes as extractor) ───────────────────────────────
max_channels = len(channels)
asymmetry_pairs = [('C3','C4'), ('FP1','FP2'), ('F3','F4')]
coherence_pairs = [('C3','C4'), ('FP1','FP2')]

def compute_psd(epochs, channel_indices):
    n_epochs = epochs.shape[0]
    out = np.zeros((n_epochs, max_channels * len(freq_bands)), dtype=np.float32)
    for e in range(n_epochs):
        for ch_idx in channel_indices:
            freqs, pxx = welch(epochs[e, ch_idx], fs=sfreq, nperseg=256, noverlap=128)
            for b_idx, (_, (lo, hi)) in enumerate(freq_bands.items()):
                idx = (freqs >= lo) & (freqs < hi)
                out[e, ch_idx*len(freq_bands)+b_idx] = np.sum(pxx[idx]) if idx.any() else 0.0
    return np.nan_to_num(out, nan=0.0)

def compute_asymmetry(epochs):
    n_epochs = epochs.shape[0]
    out = np.zeros((n_epochs, len(asymmetry_pairs)*len(freq_bands)), dtype=np.float32)
    for e in range(n_epochs):
        psd_cache = {}
        for p_idx, (L, R) in enumerate(asymmetry_pairs):
            if L in channels and R in channels:
                li, ri = channels.index(L), channels.index(R)
                if (e, li) not in psd_cache:
                    f, lpsd = welch(epochs[e, li], fs=sfreq, nperseg=256, noverlap=128); psd_cache[(e, li)] = (f, lpsd)
                else: f, lpsd = psd_cache[(e, li)]
                if (e, ri) not in psd_cache:
                    _, rpsd = welch(epochs[e, ri], fs=sfreq, nperseg=256, noverlap=128); psd_cache[(e, ri)] = (f, rpsd)
                else: _, rpsd = psd_cache[(e, ri)]
                for b_idx, (_, (lo, hi)) in enumerate(freq_bands.items()):
                    idx = (f >= lo) & (f < hi)
                    lp = np.sum(lpsd[idx]) if idx.any() else 0.0
                    rp = np.sum(rpsd[idx]) if idx.any() else 0.0
                    out[e, p_idx*len(freq_bands)+b_idx] = (lp - rp) / (lp + rp + 1e-10)
    return np.nan_to_num(out, nan=0.0)

def compute_coherence(epochs):
    n_epochs = epochs.shape[0]
    out = np.zeros((n_epochs, len(coherence_pairs)*len(freq_bands)), dtype=np.float32)
    for e in range(n_epochs):
        row = []
        for a,b in coherence_pairs:
            if a in channels and b in channels:
                i1, i2 = channels.index(a), channels.index(b)
                f, coh = coherence(epochs[e, i1], epochs[e, i2], fs=sfreq, nperseg=256, noverlap=128)
                for _, (lo, hi) in freq_bands.items():
                    idx = (f >= lo) & (f < hi)
                    row.append(float(np.mean(coh[idx])) if idx.any() else 0.0)
            else:
                row.extend([0.0]*len(freq_bands))
        out[e] = row
    return np.nan_to_num(out, nan=0.0)

def compute_time_domain(epochs, channel_indices):
    n_epochs = epochs.shape[0]
    out = np.zeros((n_epochs, max_channels*5), dtype=np.float32)
    for e in range(n_epochs):
        for ch in channel_indices:
            sig = epochs[e, ch]
            base = [np.mean(sig), np.var(sig), skew(sig), kurtosis(sig), entropy(np.abs(sig)+1e-10)]
            out[e, ch*5:(ch+1)*5] = base
    return np.nan_to_num(out, nan=0.0)

def compute_wavelet(epochs, channel_indices):
    n_epochs = epochs.shape[0]
    out = np.zeros((n_epochs, max_channels*4), dtype=np.float32)
    scales = np.arange(1,129)
    for e in range(n_epochs):
        for ch in channel_indices:
            coeffs, _ = cwt(epochs[e, ch], scales=scales, wavelet='morl')
            vals = [np.mean(coeffs), np.std(coeffs), np.max(coeffs), np.min(coeffs)]
            out[e, ch*4:(ch+1)*4] = vals
    return np.nan_to_num(out, nan=0.0)

def compute_band_ratios(psd_features, n_channels):
    n_epochs = psd_features.shape[0]
    bands = list(freq_bands.keys())
    rcount = (len(bands)*(len(bands)-1))//2
    out = np.zeros((n_epochs, n_channels*rcount), dtype=np.float32)
    for e in range(n_epochs):
        dest = []
        for ch in range(n_channels):
            band_p = psd_features[e, ch*len(bands):(ch+1)*len(bands)]
            for i in range(len(bands)):
                for j in range(i+1, len(bands)):
                    dest.append(band_p[i] / (band_p[j] + 1e-10))
        out[e] = dest
    return np.nan_to_num(out, nan=0.0)

# ─── BIOMARKER EXTRAS (pdBSI, entropy, CWT summaries) ─────────────────────────
import numpy as _np
from scipy.signal import welch as _welch
from scipy.stats import skew as _skew, kurtosis as _kurt
import pywt as _pywt

_THESIS_FS_DEFAULT = 250

def thesis_welch_psd(x, fs=_THESIS_FS_DEFAULT):
    nper = int(2*fs); nover = int(1*fs)
    f, pxx = _welch(x, fs=fs, window="hamming", nperseg=nper, noverlap=nover, axis=-1)
    return f, pxx

_THESIS_ASYM_PAIRS = [("FP1","FP2"), ("F7","F8"), ("F3","F4"), ("C3","C4"),
                      ("P3","P4"), ("T3","T4"), ("T5","T6"), ("O1","O2")]

def _thesis_norm_name(s: str) -> str:
    s = (s or "").upper().strip()
    for pref in ("EEG ", "EEG"):
        if s.startswith(pref): s = s[len(pref):]
    for suf in ("-REF", "-LE", "-AVG"):
        if s.endswith(suf): s = s[: -len(suf)]
    return s

def _find_idx_map(ch_names):
    lut = { _thesis_norm_name(c): i for i, c in enumerate(ch_names) }
    pairs = []
    for a, b in _THESIS_ASYM_PAIRS:
        au, bu = _thesis_norm_name(a), _thesis_norm_name(b)
        if au in lut and bu in lut: pairs.append((lut[au], lut[bu]))
    return pairs

def thesis_sample_entropy(x, m=2, r=None):
    x = _np.asarray(x, dtype=float)
    r = 0.2 * _np.std(x) if r is None else r
    n = x.size
    if n <= m + 1: return _np.nan
    def _phi(mm):
        xm = _np.array([x[i:i+mm] for i in range(n-mm+1)])
        C = _np.sum(_np.max(_np.abs(xm[:, None, :] - xm[None, :, :]), axis=2) <= r, axis=1) - 1
        denom = (n - mm + 1) * (n - mm)
        return _np.sum(C) / max(denom, 1)
    a = _phi(m); b = _phi(m+1)
    if a <= 0 or b <= 0: return _np.nan
    return -_np.log(b / a)

def thesis_pdbsi(left_psd, right_psd, freqs, fmin=1.0, fmax=40.0):
    idx = (freqs >= fmin) & (freqs <= fmax)
    L = left_psd[..., idx]; R = right_psd[..., idx]
    return _np.mean(_np.abs(L - R) / _np.maximum(L + R, 1e-12), axis=-1)

def thesis_morlet_stats(x, fs=_THESIS_FS_DEFAULT, n_scales=32):
    try:
        scales = _np.logspace(_np.log10(1), _np.log10(128), n_scales)
        coef, _ = _pywt.cwt(x, scales, 'morl', sampling_period=1.0/fs)
        return {
            "cwt_mean": _np.mean(_np.abs(coef)),
            "cwt_std":  _np.std(coef),
            "cwt_max":  _np.max(_np.abs(coef)),
            "cwt_min":  _np.min(_np.abs(coef)),
            "cwt_skew": _skew(_np.ravel(_np.real(coef)), bias=False),
            "cwt_kurt": _kurt(_np.ravel(_np.real(coef)), fisher=True, bias=False),
        }
    except Exception:
        return {k:_np.nan for k in ["cwt_mean","cwt_std","cwt_max","cwt_min","cwt_skew","cwt_kurt"]}

def thesis_augment_features(epochs, fs, ch_names, base):
    data = epochs if isinstance(epochs, _np.ndarray) else _np.asarray(epochs)
    n_ep, n_ch, _ = data.shape
    f, psd = thesis_welch_psd(data, fs=fs)

    pairs = _find_idx_map(ch_names)
    pdbsi_vals = []
    for i in range(n_ep):
        vals = []
        for li, ri in pairs:
            vals.append(thesis_pdbsi(psd[i, li:li+1, :], psd[i, ri:ri+1, :], f))
        pdbsi_vals.append(_np.mean(vals) if vals else _np.nan)

    sampen = _np.array([
        _np.nanmean([thesis_sample_entropy(data[i, ch]) for ch in range(n_ch)])
        for i in range(n_ep)
    ])

    cwt_mean=[]; cwt_std=[]; cwt_max=[]; cwt_min=[]; cwt_skew=[]; cwt_kurt=[]
    for i in range(n_ep):
        stats = [thesis_morlet_stats(data[i, ch], fs=fs) for ch in range(n_ch)]
        cwt_mean.append(_np.nanmean([s["cwt_mean"] for s in stats]))
        cwt_std.append(_np.nanmean([s["cwt_std"] for s in stats]))
        cwt_max.append(_np.nanmean([s["cwt_max"] for s in stats]))
        cwt_min.append(_np.nanmean([s["cwt_min"] for s in stats]))
        cwt_skew.append(_np.nanmean([s["cwt_skew"] for s in stats]))
        cwt_kurt.append(_np.nanmean([s["cwt_kurt"] for s in stats]))

    extras = _np.column_stack([
        _np.array(pdbsi_vals).reshape(-1),
        sampen.reshape(-1),
        _np.array(cwt_mean).reshape(-1),
        _np.array(cwt_std).reshape(-1),
        _np.array(cwt_max).reshape(-1),
        _np.array(cwt_min).reshape(-1),
        _np.array(cwt_skew).reshape(-1),
        _np.array(cwt_kurt).reshape(-1),
    ])
    base_arr = _np.asarray(base)
    return _np.concatenate([base_arr, extras], axis=1)

def thesis_feature_name_extras():
    return ["pdbsi_1_40","sampen_m2_r02","cwt_mean","cwt_std","cwt_max","cwt_min","cwt_skew","cwt_kurt"]

# ─── EVAL FEATURE EXTRACTION ───────────────────────────────────────────────────
def load_and_extract_eval_features(eval_dirs):
    Xt, yt, wt = [], [], []
    for d in eval_dirs:
        fns = sorted(glob.glob(os.path.join(d, "*_epochs.npy")))
        logger.info(f"Found {len(fns)} eval files in {d}")
        for fn in tqdm(fns, desc=f"Eval[{os.path.basename(d)}] files", unit="file"):
            t0_file = time.perf_counter()
            arr = np.load(fn, mmap_mode='r').astype(np.float32)
            windows = arr[np.newaxis, ...] if arr.ndim == 2 else arr
            fname = os.path.basename(fn)

            if 'stroke' in fn.lower():
                m = re.search(r'(sub-\d+)', fn.lower())
                subj = m.group(1) if m else fname.replace('.npy','')
                label = 1; weight = get_feature_weights(subj)
            else:
                subj = None; label = 0; weight = 1.0

            n_channels = windows.shape[1]
            if n_channels < min_channels:
                logger.warning(f"Skipping {fn} due to fewer channels ({n_channels})")
                continue

            channel_indices = list(range(min(n_channels, len(channels))))

            # Stage timers
            t1 = time.perf_counter(); psd_features = compute_psd(windows, channel_indices)
            t_psd = time.perf_counter() - t1

            t1 = time.perf_counter(); asymmetry_features = compute_asymmetry(windows)
            t_asym = time.perf_counter() - t1

            t1 = time.perf_counter(); coherence_features = compute_coherence(windows)
            t_coh = time.perf_counter() - t1

            t1 = time.perf_counter(); time_features = compute_time_domain(windows, channel_indices)
            t_time = time.perf_counter() - t1

            t1 = time.perf_counter(); wavelet_features = compute_wavelet(windows, channel_indices)
            t_wav = time.perf_counter() - t1

            t1 = time.perf_counter(); band_ratio_features = compute_band_ratios(psd_features, n_channels)
            t_ratio = time.perf_counter() - t1

            features = np.concatenate(
                [psd_features, asymmetry_features, coherence_features,
                 time_features, wavelet_features, band_ratio_features],
                axis=1
            )

            # Add the 8 biomarker extras so eval matches training columns
            try:
                fs = int(round(windows.shape[-1] / 10.0)) if windows.shape[-1] > 0 else 250
            except Exception:
                fs = 250
            ch_names = channels[:n_channels]
            features = thesis_augment_features(windows, fs, ch_names, features)

            features = np.nan_to_num(features, nan=0.0)

            Xt.extend(features)
            yt.extend([label] * len(windows))
            wt.extend([weight] * len(windows))

            t_file = time.perf_counter() - t0_file
            logger.info(f"[EVAL] {os.path.basename(fn)} tot={t_file:.1f}s | "
                        f"psd={t_psd:.1f}s asym={t_asym:.1f}s coh={t_coh:.1f}s "
                        f"time={t_time:.1f}s wav={t_wav:.1f}s ratio={t_ratio:.1f}s | feats={features.shape[1]}")

    Xt_arr = np.stack(Xt, axis=0)
    yt_arr = np.array(yt, dtype=np.int32)
    wt_arr = np.array(wt, dtype=np.float32)
    logger.info(f"EVAL FEATURES: {Xt_arr.shape[0]} samples, {Xt_arr.shape[1]} features")
    return Xt_arr, yt_arr, wt_arr

# ─── FEATURE NAMES ─────────────────────────────────────────────────────────────
def generate_feature_names():
    names = []
    bands = list(freq_bands.keys())
    # PSD
    for ch in channels:
        for band in bands: names.append(f'PSD_{ch}_{band}')
    # Asymmetry
    for a,b in asymmetry_pairs:
        for band in bands: names.append(f'ASYM_{a}_{b}_{band}')
    # Coherence
    for a,b in coherence_pairs:
        for band in bands: names.append(f'COH_{a}_{b}_{band}')
    # Time
    for ch in channels:
        names.extend([f'TIME_{ch}_mean', f'TIME_{ch}_var', f'TIME_{ch}_skew', f'TIME_{ch}_kurt', f'TIME_{ch}_entropy'])
    # Wavelet
    for ch in channels:
        names.extend([f'WAV_{ch}_mean', f'WAV_{ch}_std', f'WAV_{ch}_max', f'WAV_{ch}_min'])
    # Ratios
    bands_list = list(freq_bands.keys())
    for ch in channels:
        for i in range(len(bands_list)):
            for j in range(i+1, len(bands_list)):
                names.append(f'RATIO_{ch}_{bands_list[i]}_over_{bands_list[j]}')
    # Extras (8)
    names += thesis_feature_name_extras()
    return names

# ─── PLOTS ─────────────────────────────────────────────────────────────────────
def plot_confusion_and_curves(y_true, y_prob, set_id, outdir=PLOTS_DIR):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Set {set_id} @0.5"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"cm_set_{set_id}.png")); plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4,3)); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--', alpha=0.6)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - Set {set_id}"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"roc_set_{set_id}.png")); plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = np.trapz(prec[::-1], rec[::-1])
    plt.figure(figsize=(4,3)); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - Set {set_id} (AUC={auc_pr:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"pr_set_{set_id}.png")); plt.close()

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # Load TRAIN features (from extractor script)
    X_train_all = np.load(os.path.join(FEATURE_DIR, "features.npy"))
    y_train_all = np.load(os.path.join(FEATURE_DIR, "labels.npy"))
    w_train_all = np.load(os.path.join(FEATURE_DIR, "weights.npy"))
    logger.info(f"TRAIN: {X_train_all.shape[0]} samples, {X_train_all.shape[1]} features")

    # Eval features (compute from eval epoch npys)
    X_test, y_test, w_test = load_and_extract_eval_features([EVAL_NORMAL, EVAL_STROKE])
    X_test = np.nan_to_num(X_test, nan=0.0)

    y_train_bin = (y_train_all >= 0.5).astype(int)
    y_test_bin  = (y_test      >= 0.5).astype(int)
    logger.info(f"Label balance — train pos={y_train_bin.sum()}/{y_train_bin.size}, "
                f"eval pos={y_test_bin.sum()}/{y_test_bin.size}")

    # Names
    feature_names = generate_feature_names()
    if len(feature_names) != X_train_all.shape[1]:
        missing = X_train_all.shape[1] - len(feature_names)
        logger.warning(f"Feature-name mismatch: have {len(feature_names)} names for {X_train_all.shape[1]} columns; padding {missing}.")
        if missing > 0: feature_names += [f"EXTRA_{i}" for i in range(missing)]
        else: feature_names = feature_names[:X_train_all.shape[1]]
    logger.info(f"Generated {len(feature_names)} feature names")

    # Param sets
    param_sets = [
        {'max_depth': 3, 'eta': 0.1, 'scale_pos_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0, 'gamma': 0.0, 'num_boost_round': 200},
        {'max_depth': 4, 'eta': 0.05, 'scale_pos_weight': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.0, 'num_boost_round': 400},
        {'max_depth': 5, 'eta': 0.03, 'scale_pos_weight': 6, 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.0, 'num_boost_round': 600},
    ]

    best_auc = -1.0
    best_model = None   
    best_set_id = None
    best_feats = None
    dtrain_best = None
    Xsel_best = None
    best_selector_filter = None
    best_selector_wrapper = None
    best_thr_overall = None
    best_metrics = None

    for set_id, ps in enumerate(param_sets, 1):
        params = {
            'max_depth': ps['max_depth'],
            'eta': ps['eta'],
            'subsample': ps['subsample'],
            'colsample_bytree': ps['colsample_bytree'],
            'gamma': ps['gamma'],
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'nthread': 0,
            'scale_pos_weight': ps['scale_pos_weight'],  # IMPORTANT: apply class weight
        }
        logger.info(f"Parameter Set {set_id}: {params}")

        # Filter: Mutual Info top 200
        logger.info(f"Set {set_id}: SelectKBest(mutual_info) → top 200 ...")
        selector_filter = SelectKBest(mutual_info_classif, k=200)
        X_train_filtered = selector_filter.fit_transform(X_train_all, y_train_bin)
        X_test_filtered  = selector_filter.transform(X_test)
        idx200 = selector_filter.get_support(indices=True)

        # Wrapper: RFE to 50
        logger.info(f"Set {set_id}: RFE(XGBClassifier) → top 50 (this step can be slow)...")
        est = xgb.XGBClassifier(
            max_depth=params['max_depth'],
            learning_rate=params['eta'],  # alias
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            n_estimators=200,  # small internal model for RFE
            random_state=42,
            scale_pos_weight=params['scale_pos_weight'],
            n_jobs=0
        )
        selector_wrapper = RFE(est, n_features_to_select=50, step=0.25)
        X_train_selected = selector_wrapper.fit_transform(X_train_filtered, y_train_bin)
        X_test_selected  = selector_wrapper.transform(X_test_filtered)

        # Map selected indices → original names
        support50_in200 = selector_wrapper.support_
        orig_indices50 = idx200[support50_in200]
        top_features = [feature_names[i] for i in orig_indices50 if i < len(feature_names)]
        logger.info(f"Set {set_id}: Selected {len(top_features)} features: {top_features}")

        # Optional grouped CV (only if groups.npy exists)
        try:
            groups_path = os.path.join(FEATURE_DIR, "groups.npy")
            if os.path.exists(groups_path):
                groups = np.load(groups_path)
                metrics, _ = thesis_grouped_cv_train(X_train_selected, y_train_bin, groups)
                for m in metrics:
                    logger.info(f"[GroupedCV] fold={m['fold']} acc={m['acc']:.3f} prec={m['prec']:.3f} "
                                f"rec={m['rec']:.3f} f1={m['f1']:.3f} roc={m.get('roc', float('nan')):.3f} "
                                f"best_iter={m['best_iteration_']}")
        except Exception as e:
            logger.warning(f"[GroupedCV] skipped: {e}")

        # Train final booster on selected features
        dtrain = xgb.DMatrix(X_train_selected, label=y_train_bin, weight=w_train_all)
        dtest  = xgb.DMatrix(X_test_selected,  label=y_test_bin,  weight=w_test)
        logger.info(f"Set {set_id}: Training XGBoost for {ps['num_boost_round']} rounds...")
        bst = xgb.train(params, dtrain, num_boost_round=ps['num_boost_round'], evals=[(dtrain, 'train')], verbose_eval=False)

        # Predict & choose best threshold (max F1)
        y_prob = bst.predict(dtest)
        thr_grid = np.linspace(0.05, 0.95, 19)
        f1s = []
        for thr in thr_grid:
            f1s.append(f1_score(y_test_bin, (y_prob >= thr).astype(int), zero_division=0))
        best_thr = float(thr_grid[int(np.argmax(f1s))])

        # Metrics @ best_thr
        y_pred = (y_prob >= best_thr).astype(int)
        acc = accuracy_score(y_test_bin, y_pred)
        prec = precision_score(y_test_bin, y_pred, zero_division=0)
        rec = recall_score(y_test_bin, y_pred, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob); roc_auc = auc(fpr, tpr)
        prec_c, rec_c, _ = precision_recall_curve(y_test_bin, y_prob)
        auc_pr = np.trapz(prec_c[::-1], rec_c[::-1])

        logger.info(f"Set {set_id}: THR={best_thr:.2f} ACC={acc:.3f} PREC={prec:.3f} REC={rec:.3f} "
                    f"F1={f1:.3f} AUCROC={roc_auc:.3f} AUCPR={auc_pr:.3f}")

        # Track best by ROC AUC (you can swap to AUCPR if preferred)
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = bst
            best_set_id = set_id
            best_feats = top_features
            dtrain_best = dtrain
            Xsel_best = X_train_selected
            best_selector_filter = selector_filter
            best_selector_wrapper = selector_wrapper
            best_thr_overall = best_thr
            best_metrics = dict(acc=acc, prec=prec, rec=rec, f1=f1,
                        auc_roc=roc_auc, auc_pr=auc_pr, thr=best_thr)

        # Plots at default 0.5 for apples-to-apples (and you logged the best_thr above)
        plot_confusion_and_curves(y_test_bin, y_prob, set_id)
    # save booster (native XGBoost)
    model_path = os.path.join(MODEL_DIR, f"xgb_set{best_set_id}.json")
    best_model.save_model(model_path)

    # save fitted selectors
    joblib.dump(best_selector_filter, os.path.join(MODEL_DIR, f"selector_filter_set{best_set_id}.joblib"))
    joblib.dump(best_selector_wrapper, os.path.join(MODEL_DIR, f"selector_wrapper_set{best_set_id}.joblib"))

    # save metadata (features, threshold, params, channel order)
    bundle = {
        "best_set_id": best_set_id,
        "threshold": best_thr_overall,
        "selected_feature_names": best_feats,
        "channels": channels,
        "sfreq": sfreq,
        "metrics": best_metrics,
    }
    with open(os.path.join(MODEL_DIR, f"bundle_set{best_set_id}.json"), "w") as f:
        json.dump(bundle, f, indent=2)
    logger.info(f"Saved model bundle to {MODEL_DIR}")

    logger.info(f"Best Set: {best_set_id} with AUCROC={best_auc:.3f}")

    # SHAP on best model (optional)
    try:
        logger.info("Computing SHAP summary (can be slow)...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(Xsel_best)
        shap.summary_plot(shap_values, Xsel_best, feature_names=best_feats, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"shap_summary_set_{best_set_id}.png")); plt.close()
    except Exception as e:
        logger.warning(f"SHAP summary skipped: {e}")

    # t-SNE visual (optional)
    try:
        logger.info("Running t-SNE embedding (slow; seconds–minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_emb = tsne.fit_transform(Xsel_best)
        plt.figure(figsize=(4,3))
        plt.scatter(X_emb[:,0], X_emb[:,1], c=(np.array(y_train_all)>=0.5).astype(int), cmap='coolwarm', s=8, alpha=0.8)
        plt.title("t-SNE of selected features (train)")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"tsne_set_{best_set_id}.png")); plt.close()
    except Exception as e:
        logger.warning(f"TSNE skipped: {e}")

# ─── OPTIONAL: grouped CV helper (only runs if groups.npy present) ─────────────
from typing import List, Tuple
from sklearn.model_selection import GroupKFold
def thesis_grouped_cv_train(X, y, groups,
                            n_splits: int = 5,
                            scale_pos_weight: float | None = None,
                            n_estimators: int = 1000,
                            learning_rate: float = 0.05,
                            max_depth: int = 4,
                            subsample: float = 0.9,
                            colsample_bytree: float = 0.9,
                            early_stopping_rounds: int = 100):
    X = np.asarray(X); y = (np.asarray(y).reshape(-1) >= 0.5).astype(int)
    if groups is None: raise ValueError("groups is required")
    if scale_pos_weight is None:
        pos = max(y.sum(), 1); neg = max((y==0).sum(), 1)
        scale_pos_weight = neg/pos

    gkf = GroupKFold(n_splits=n_splits)
    metrics = []; models = []
    fold = 0
    for tr, va in gkf.split(X, y, groups=groups):
        fold += 1
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=subsample, colsample_bytree=colsample_bytree,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            reg_lambda=1.0, random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=0
        )
        clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False,
                early_stopping_rounds=early_stopping_rounds)
        proba = clf.predict_proba(X[va])[:,1]; yhat = (proba>=0.5).astype(int)
        try:
            from sklearn.metrics import roc_auc_score
            roc = float(roc_auc_score(y[va], proba))
        except Exception:
            roc = float("nan")
        m = dict(
            fold=fold,
            acc=float(accuracy_score(y[va], yhat)),
            prec=float(precision_score(y[va], yhat, zero_division=0)),
            rec=float(recall_score(y[va], yhat, zero_division=0)),
            f1=float(f1_score(y[va], yhat, zero_division=0)),
            roc=roc,
            best_iteration_=int(getattr(clf, "best_iteration_", 0) or 0),
        )
        metrics.append(m); models.append(clf)
    return metrics, models

# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
