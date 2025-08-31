# Updated 4_extract_features.py
# Changes: minimal/safe — infer real fs, thesis Welch/Coherence (2s Hamming, 50% overlap),
# case-safe channel mapping for asymmetry, scale once at end, optional thesis augment (try/except).

import numpy as np
import os
import pandas as pd
import logging
from datetime import datetime
from scipy.signal import welch, coherence
from scipy.stats import entropy, skew, kurtosis
from pywt import cwt
from psutil import Process
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Logging ----------------
log_dir = r"C:/UoM Sem-2/Research_work/src/logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"extract_features_{timestamp}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- Paths ----------------
epoch_dir = r"C:/UoM Sem-2/Research_work/src/epochs"
feature_dir = r"C:/UoM Sem-2/Research_work/src/features"
plot_dir = r"C:/UoM Sem-2/Research_work/src/plots/features"
os.makedirs(feature_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---------------- Metadata ----------------
participants_path = "C:/UoM Sem-2/Research_work/Datasets/Dataset/FIgshare/metadata/participants.tsv"
use_metadata = True
try:
    participants = pd.read_csv(participants_path, sep='\t')
    participant_dict = participants.set_index('Participant_ID').to_dict('index')
    logger.info("Loaded participants.tsv with stroke metadata")
except Exception as e:
    logger.error(f"Failed to load participants.tsv: {str(e)}. Using default weights.")
    participant_dict = {}
    use_metadata = False

# ---------------- Config ----------------
channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
freq_bands = {'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 100.0)}
sfreq = 256.0   # default; will be overridden per file (see below)
min_channels = 10
asymmetry_pairs = [('C3', 'C4'), ('FP1', 'FP2'), ('F3', 'F4')]
coherence_pairs = [('C3', 'C4'), ('FP1', 'FP2')]

# Only used to decide where to scale/PCA — preserves your original lines without deleting them
_THESIS_SCALE_PER_FILE = False
_THESIS_PCA_PER_FILE = False  # keep PCA (if enabled) to run once after concatenation

# ---------------- Helpers ----------------
def get_memory_usage():
    process = Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def get_feature_weights(participant_id):
    if not use_metadata or participant_id not in participant_dict:
        logger.info(f"No metadata for participant {participant_id}, using default weight 1.0")
        return 1.0
    nihss = participant_dict[participant_id]['NIHSS']
    stroke_location = str(participant_dict[participant_id]['StrokeLocation']).lower()
    weight = 1.0 + (nihss / 11.0)
    if 'pons' in stroke_location or 'basal ganglia' in stroke_location:
        weight *= 1.2
    return weight

def _norm(s: str) -> str:
    s = (s or "").upper().strip()
    for p in ("EEG ", "EEG"):
        if s.startswith(p): s = s[len(p):]
    for q in ("-REF","-LE","-AVG"):
        if s.endswith(q): s = s[:-len(q)]
    return s

def compute_psd(epochs, channel_indices):
    # Uses global sfreq (overridden per file inside extract_features via `global sfreq`)
    win = int(2 * sfreq)  # 2 s
    ovr = int(1 * sfreq)  # 50% overlap
    n_epochs, _, _ = epochs.shape
    psd_features = []
    for epoch in range(n_epochs):
        epoch_psd = []
        for ch_idx in channel_indices:
            freqs, psd = welch(epochs[epoch, ch_idx], fs=sfreq, window="hamming", nperseg=win, noverlap=ovr)
            band_psd = []
            for band, (low, high) in freq_bands.items():
                idx_band = np.where((freqs >= low) & (freqs < high))[0]
                band_power = np.sum(psd[idx_band]) if idx_band.size > 0 else 0.0
                band_psd.append(band_power)
            epoch_psd.extend(band_psd)
        psd_features.append(epoch_psd)
    return np.array(psd_features)

def compute_asymmetry(epochs, channel_indices_dict, asym_pairs_upper):
    # Uses Welch params same as compute_psd
    win = int(2 * sfreq)
    ovr = int(1 * sfreq)
    n_epochs, n_channels, _ = epochs.shape
    asymmetry_features = []
    psd_cache = {}  # Cache PSD to avoid recomputing
    for epoch in range(n_epochs):
        epoch_asymmetry = []
        for left_ch, right_ch in asym_pairs_upper:
            if left_ch in channel_indices_dict and right_ch in channel_indices_dict:
                left_idx, right_idx = channel_indices_dict[left_ch], channel_indices_dict[right_ch]
                key_left = (epoch, left_idx)
                key_right = (epoch, right_idx)
                if key_left not in psd_cache:
                    freqs, left_psd = welch(epochs[epoch, left_idx], fs=sfreq, window="hamming", nperseg=win, noverlap=ovr)
                    psd_cache[key_left] = (freqs, left_psd)
                else:
                    freqs, left_psd = psd_cache[key_left]
                if key_right not in psd_cache:
                    _, right_psd = welch(epochs[epoch, right_idx], fs=sfreq, window="hamming", nperseg=win, noverlap=ovr)
                    psd_cache[key_right] = (freqs, right_psd)
                else:
                    _, right_psd = psd_cache[key_right]
                for band, (low, high) in freq_bands.items():
                    idx_band = np.where((freqs >= low) & (freqs < high))[0]
                    left_power = np.sum(left_psd[idx_band]) if idx_band.size > 0 else 0.0
                    right_power = np.sum(right_psd[idx_band]) if idx_band.size > 0 else 0.0
                    asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
                    epoch_asymmetry.append(asymmetry)
            else:
                logging.warning(f"Channel pair ({left_ch}, {right_ch}) not found in epoch file")
                epoch_asymmetry.extend([0.0] * len(freq_bands))
        asymmetry_features.append(epoch_asymmetry)
    return np.array(asymmetry_features)

def compute_coherence(epochs, channel_indices_dict, coh_pairs_upper):
    win = int(2 * sfreq)
    ovr = int(1 * sfreq)
    n_epochs, n_channels, _ = epochs.shape
    coherence_features = []
    for epoch in range(n_epochs):
        epoch_coherence = []
        for ch1, ch2 in coh_pairs_upper:
            if ch1 in channel_indices_dict and ch2 in channel_indices_dict:
                idx1, idx2 = channel_indices_dict[ch1], channel_indices_dict[ch2]
                freqs, coh = coherence(epochs[epoch, idx1], epochs[epoch, idx2], fs=sfreq, window="hamming", nperseg=win, noverlap=ovr)
                for band, (low, high) in freq_bands.items():
                    idx_band = np.where((freqs >= low) & (freqs < high))[0]
                    band_coherence = np.mean(coh[idx_band]) if idx_band.size > 0 else 0.0
                    epoch_coherence.append(band_coherence)
            else:
                logging.warning(f"Channel pair ({ch1}, {ch2}) not found in epoch file")
                epoch_coherence.extend([0.0] * len(freq_bands))
        coherence_features.append(epoch_coherence)
    return np.array(coherence_features)

def compute_time_domain(epochs, channel_indices):
    n_epochs, _, _ = epochs.shape
    time_features = []
    for epoch in range(n_epochs):
        epoch_time = []
        for ch_idx in channel_indices:
            sig = epochs[epoch, ch_idx]
            epoch_time.extend([np.mean(sig), np.var(sig), skew(sig), kurtosis(sig), entropy(np.abs(sig) + 1e-10)])
        time_features.append(epoch_time)
    return np.array(time_features)

def compute_wavelet(epochs, channel_indices):
    n_epochs, _, _ = epochs.shape
    wavelet_features = []
    scales = np.arange(1, 129)  # 'morl' wavelet
    for epoch in range(n_epochs):
        epoch_wavelet = []
        for ch_idx in channel_indices:
            coeffs, _ = cwt(epochs[epoch, ch_idx], scales=scales, wavelet='morl')
            epoch_wavelet.extend([np.mean(coeffs), np.std(coeffs), np.max(coeffs), np.min(coeffs)])
        wavelet_features.append(epoch_wavelet)
    return np.array(wavelet_features)
def compute_band_ratios(psd_features, n_channels):
    n_epochs = psd_features.shape[0]
    band_ratio_features = []
    bands_list = list(freq_bands.keys())
    for epoch in range(n_epochs):
        epoch_ratios = []
        for ch in range(n_channels):
            band_powers = psd_features[epoch, ch*len(bands_list):(ch+1)*len(bands_list)]
            for i in range(len(bands_list)):
                for j in range(i+1, len(bands_list)):
                    ratio = band_powers[i] / (band_powers[j] + 1e-10)
                    epoch_ratios.append(ratio)
        band_ratio_features.append(epoch_ratios)
    return np.array(band_ratio_features)

def validate_features(all_features, all_labels):
    logger.info("Validating extracted features...")
    df = pd.DataFrame(all_features)
    df['label'] = (all_labels > 0.5).astype(int)
    stats = df.groupby('label').mean().T
    logger.info(f"Feature means by class:\n{stats}")
    melted = df.melt(id_vars='label', var_name='feature', value_name='value')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x='feature', y='value', hue='label')
    plt.title("Feature Distributions by Class (Stroke=1, Normal=0)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "feature_dist_by_class.png"))
    plt.close()
    logger.info("Saved feature distribution plot")

# -------- optional: thesis augment helpers (safe to skip if anything fails) --------
def _thesis_sample_entropy(x, m=2, r=None):
    x = np.asarray(x, dtype=float)
    if r is None: r = 0.2 * np.std(x)
    n = x.size
    if n <= m + 1: return np.nan
    def _phi(mm):
        xm = np.array([x[i:i+mm] for i in range(n-mm+1)])
        C = np.sum(np.max(np.abs(xm[:, None, :] - xm[None, :, :]), axis=2) <= r, axis=1) - 1
        denom = (n - mm + 1) * (n - mm)
        return np.sum(C) / max(denom, 1)
    a = _phi(m); b = _phi(m+1)
    if a <= 0 or b <= 0: return np.nan
    return -np.log(b / a)

def _thesis_welch_psd_block(X, fs):
    # X: (n_ep, n_ch, n_t)
    win = int(2 * fs); ovr = int(1 * fs)
    n_ep, n_ch, _ = X.shape
    # Compute once per-epoch/channel
    f = None
    P = []
    for i in range(n_ep):
        Pi = []
        for ch in range(n_ch):
            f_i, pxx = welch(X[i, ch], fs=fs, window="hamming", nperseg=win, noverlap=ovr)
            if f is None: f = f_i
            Pi.append(pxx)
        P.append(np.stack(Pi, axis=0))
    return f, np.stack(P, axis=0)  # (n_ep, n_ch, n_f)

def _thesis_pdBSI(left_psd, right_psd, freqs, fmin=1.0, fmax=40.0):
    idx = (freqs >= fmin) & (freqs <= fmax)
    L = left_psd[..., idx]; R = right_psd[..., idx]
    num = np.abs(L - R); den = np.maximum(L + R, 1e-12)
    return np.mean(num / den, axis=-1)

def thesis_augment_features(epochs_nd, fs, ch_names, base_array):
    """
    Adds: pdBSI(1–40), Sample Entropy (mean over channels), Morlet stats (mean over channels).
    Returns np.ndarray with new columns appended. Wrapped by caller with try/except.
    """
    X = epochs_nd if isinstance(epochs_nd, np.ndarray) else np.asarray(epochs_nd)
    n_ep, n_ch, _ = X.shape
    # Welch block
    freqs, psd = _thesis_welch_psd_block(X, fs)
    # Left-right pairs (robust, uppercase)
    pairs_spec = [("FP1","FP2"), ("F7","F8"), ("F3","F4"), ("C3","C4"),
                  ("P3","P4"), ("T3","T4"), ("T5","T6"), ("O1","O2")]
    # Build map from actual ch_names
    lut = {_norm(c): i for i, c in enumerate(ch_names)}
    pairs = [(lut[a], lut[b]) for a, b in pairs_spec if a in lut and b in lut]
    pdbsi = []
    for i in range(n_ep):
        vals = []
        for li, ri in pairs:
            vals.append(_thesis_pdBSI(psd[i, li:li+1, :], psd[i, ri:ri+1, :], freqs))
        pdbsi.append(np.mean(vals) if vals else np.nan)
    # Sample entropy + Morlet summaries
    import pywt
    cwt_mean=[]; cwt_std=[]; cwt_max=[]; cwt_min=[]; cwt_skew=[]; cwt_kurt=[]
    sampen=[]
    scales = np.logspace(np.log10(1), np.log10(128), 32)
    from scipy.stats import skew as _sk, kurtosis as _ku
    for i in range(n_ep):
        s_list=[]; m_list=[]
        for ch in range(n_ch):
            sig = X[i, ch]
            samp = _thesis_sample_entropy(sig)
            sampen.append(samp) if ch == n_ch - 1 else None  # append once per-epoch at end
            coef, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0/fs)
            abscoef = np.abs(coef)
            m_list.append([
                np.mean(abscoef), np.std(abscoef), np.max(abscoef), np.min(abscoef),
                _sk(abscoef.ravel(), bias=False), _ku(abscoef.ravel(), fisher=True, bias=False)
            ])
        m_arr = np.array(m_list)
        cwt_mean.append(np.nanmean(m_arr[:,0])); cwt_std.append(np.nanmean(m_arr[:,1]))
        cwt_max.append(np.nanmean(m_arr[:,2]));  cwt_min.append(np.nanmean(m_arr[:,3]))
        cwt_skew.append(np.nanmean(m_arr[:,4])); cwt_kurt.append(np.nanmean(m_arr[:,5]))
    # Build extra features (align lengths)
    sampen = np.array(sampen).reshape(n_ep, )  # one per epoch (mean over channels happened via append logic)
    extra = np.column_stack([
        np.array(pdbsi).reshape(-1),
        sampen,
        np.array(cwt_mean).reshape(-1),
        np.array(cwt_std).reshape(-1),
        np.array(cwt_max).reshape(-1),
        np.array(cwt_min).reshape(-1),
        np.array(cwt_skew).reshape(-1),
        np.array(cwt_kurt).reshape(-1),
    ])
    return np.concatenate([base_array, extra], axis=1)

# ---------------- Main extraction ----------------
def extract_features(synthetic_ratio=0.3, apply_pca=False, pca_components=50):
    logger.info("Starting feature extraction")
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")

    epoch_files = []
    synthetic_files = []
    for subset in ['train']:
        for group in ['stroke', 'normal']:
            subdir = os.path.join(epoch_dir, subset, group)
            if os.path.exists(subdir):
                epoch_files.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('_epochs.npy') and 'synthetic' not in f])
            subdir_synthetic = os.path.join(epoch_dir, 'synthetic', group)
            if os.path.exists(subdir_synthetic):
                synthetic_files.extend([os.path.join(subdir_synthetic, f) for f in os.listdir(subdir_synthetic) if f.endswith('_epochs.npy')])

    total_files = len(epoch_files) + len(synthetic_files)
    if total_files > 0:
        n_synthetic = int(synthetic_ratio * total_files)
        if len(synthetic_files) > 0 and n_synthetic > 0:
            synthetic_files = np.random.choice(synthetic_files, min(n_synthetic, len(synthetic_files)), replace=False).tolist()
        all_files = epoch_files + synthetic_files
        logger.info(f"Processing {len(all_files)} files: {len(epoch_files)} real, {len(synthetic_files)} synthetic ({synthetic_ratio:.2%} synthetic)")
    else:
        all_files = epoch_files
        logger.warning("No synthetic files found, using only real data")

    all_features = []
    all_labels = []
    all_weights = []

    for epoch_file in all_files:
        try:
            is_synthetic = 'synthetic' in epoch_file.lower()
            logger.info(f"Processing {epoch_file} {'(synthetic)' if is_synthetic else ''}")

            epochs = np.load(epoch_file)  # (n_ep, n_ch, n_t)

            # --- infer real sampling rate; override module-level sfreq for downstream fns ---
            global sfreq
            try:
                sfreq = int(round(epochs.shape[-1] / 10.0))  # assuming 10 s epochs
            except Exception:
                sfreq = 250
            # -------------------------------------------------------------------------------

            risk_file = epoch_file.replace('_epochs.npy', '_risk.npy')
            risk_scores = np.load(risk_file)

            participant_id = os.path.basename(epoch_file).split('_')[2] if 'stroke' in epoch_file.lower() and not is_synthetic else None
            if is_synthetic and 'stroke' in epoch_file.lower():
                # filename format synthetic_stroke_train_sXXXX_epochs.npy
                parts = os.path.basename(epoch_file).split('_')
                participant_id = None if len(parts) < 4 else None  # synthetic has no subject metadata here

            if 'normal' in epoch_file.lower():
                participant_id = None
                logger.info(f"Non-stroke file {epoch_file}, using default weight")

            n_channels = epochs.shape[1]
            if n_channels < min_channels:
                logger.error(f"Too few channels ({n_channels}) in {epoch_file}, skipping")
                continue

            # Build case-safe channel index mapping using actual order if available
            ch_names_path = epoch_file.replace('_epochs.npy', '_ch_names.txt')
            if os.path.exists(ch_names_path):
                with open(ch_names_path, 'r') as f:
                    ch_names = [ln.strip() for ln in f if ln.strip()]
            else:
                ch_names = channels[:n_channels]  # fallback template

            channel_indices = list(range(n_channels))
            channel_indices_dict = {_norm(ch_names[i]): i for i in range(len(ch_names))}
            asym_pairs_upper = [(_norm(l), _norm(r)) for (l, r) in asymmetry_pairs]
            coh_pairs_upper  = [(_norm(l), _norm(r)) for (l, r) in coherence_pairs]

            # ---- features ----
            psd_features = compute_psd(epochs, channel_indices)
            asymmetry_features = compute_asymmetry(epochs, channel_indices_dict, asym_pairs_upper)
            coherence_features = compute_coherence(epochs, channel_indices_dict, coh_pairs_upper)
            time_features = compute_time_domain(epochs, channel_indices)
            wavelet_features = compute_wavelet(epochs, channel_indices)
            band_ratio_features = compute_band_ratios(psd_features, len(channel_indices))

            logger.info(f"PSD: {psd_features.shape}, Asym: {asymmetry_features.shape}, Coh: {coherence_features.shape}, "
                        f"Time: {time_features.shape}, Wavelet: {wavelet_features.shape}, Ratios: {band_ratio_features.shape}")

            features = np.concatenate(
                [psd_features, asymmetry_features, coherence_features, time_features, wavelet_features, band_ratio_features],
                axis=1
            )

            # --- optional thesis augment (never crash the run) ---
            try:
                ch_names_effective = [k for k, _ in sorted(channel_indices_dict.items(), key=lambda kv: kv[1])]
                features = thesis_augment_features(epochs, int(sfreq), ch_names_effective, features)
                logger.info(f"[THESIS] Augmented features shape: {np.asarray(features).shape}")
            except Exception as e:
                logger.warning(f"[THESIS] augment skipped: {e}")
            # -----------------------------------------------------

            # (keep your original PCA/scaler lines but disable per-file by default)
            if apply_pca and _THESIS_PCA_PER_FILE:
                pca = PCA(n_components=pca_components)
                features = pca.fit_transform(features)
                logger.info(f"Applied PCA (per-file): Reduced to {features.shape[1]} components")

            if _THESIS_SCALE_PER_FILE:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)

            weight = get_feature_weights(participant_id)
            all_features.append(features)
            all_labels.extend(risk_scores)
            all_weights.extend([weight] * len(risk_scores))

            logger.info(f"Extracted features for {epoch_file}: {features.shape[0]} epochs, weight: {weight:.2f}")
            logger.info(f"Memory usage: {get_memory_usage():.2f} MB")

        except Exception as e:
            logger.error(f"Error processing {epoch_file}: {str(e)}")
    # ---- after all files ----
    if all_features:
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)
        all_weights = np.array(all_weights)

        # Consistent scaling (and optional PCA) once on the full matrix
        if apply_pca and not _THESIS_PCA_PER_FILE:
            pca = PCA(n_components=pca_components)
            all_features = pca.fit_transform(all_features)
            logger.info(f"Applied PCA (global): Reduced to {all_features.shape[1]} components")

        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)

        # Validate & save
        validate_features(all_features, all_labels)
        np.save(os.path.join(feature_dir, "features.npy"), all_features)
        np.save(os.path.join(feature_dir, "labels.npy"), all_labels)
        np.save(os.path.join(feature_dir, "weights.npy"), all_weights)
        logger.info(f"Saved features: {all_features.shape}, labels: {all_labels.shape}, weights: {all_weights.shape}")
    else:
        logger.error("No features extracted. Check epoch files and logs.")

    logger.info("Feature extraction complete")

if __name__ == "__main__":
    extract_features(synthetic_ratio=0.3, apply_pca=False, pca_components=50)
