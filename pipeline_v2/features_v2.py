"""
pipeline_v2 / features_v2.py
One preprocessing + feature function used for EVERY recording (train, test, any dataset).
Output: one row of recording-level features per EDF (median over clean epochs),
plus per-epoch features saved for optional epoch-level work.

Design choices (with reasons):
- 19 standard 10-20 channels (present in both Figshare and TUH).
- Resample to 250 Hz, FIR band-pass 1-30 Hz, common average reference -> applied to the
  continuous signal BEFORE epoching (the v1 code epoched first, so CAR/resample never reached the data).
- 4 s non-overlapping epochs; reject epochs by robust peak-to-peak / flatness rules.
- Physiology features are RELATIVE powers and ratios (insensitive to amplifier gain),
  following the QEEG stroke literature: delta/alpha ratio (DAR), (delta+theta)/(alpha+beta) (DTABR),
  pairwise-derived Brain Symmetry Index (pdBSI, van Putten), signed band asymmetry per homologous pair.
- "Nuisance" features computed on the RAW signal (before filtering) to fingerprint the recording site:
  50 Hz vs 60 Hz mains-noise ratio, broadband absolute power, high-frequency slope.
"""
import numpy as np, mne
from scipy.signal import welch

mne.set_log_level("error")

CHANNELS = ['FP1','FP2','F7','F3','FZ','F4','F8','T3','C3','CZ','C4','T4','T5','P3','PZ','P4','T6','O1','O2']
PAIRS = [('FP1','FP2'),('F7','F8'),('F3','F4'),('T3','T4'),('C3','C4'),('T5','T6'),('P3','P4'),('O1','O2')]
BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
FS = 250.0
EPOCH_S = 4.0

def _norm(ch):
    c = ch.upper().strip()
    if c.startswith('EEG '): c = c[4:]
    for s in ('-REF','-LE','-AVG'):
        if c.endswith(s): c = c[:-len(s)]
    return {'T7':'T3','T8':'T4','P7':'T5','P8':'T6'}.get(c, c)

def load_standard(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    mapping, keep = {}, []
    for ch in raw.ch_names:
        n = _norm(ch)
        if n in CHANNELS and n not in mapping.values():
            mapping[ch] = n; keep.append(ch)
    missing = [c for c in CHANNELS if c not in mapping.values()]
    if missing:
        raise ValueError(f"missing channels {missing}")
    raw.pick(keep); raw.rename_channels(mapping); raw.reorder_channels(CHANNELS)
    raw.set_channel_types({c:'eeg' for c in CHANNELS})
    return raw

def nuisance_features(raw):
    """Site fingerprint on the unfiltered signal."""
    x = raw.get_data()
    fs = raw.info['sfreq']
    f, p = welch(x, fs=fs, nperseg=int(4*fs), axis=-1)
    p = p.mean(0)
    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return p[m].sum() if m.any() else np.nan
    out = {}
    out['nz_line50_vs_60_log'] = np.log10((bp(49,51)+1e-30)/(bp(59,61)+1e-30)) if fs > 125 else np.nan
    out['nz_abs_power_log'] = np.log10(bp(1,30)+1e-30)
    hf = (f >= 20) & (f <= 30)
    out['nz_hf_slope'] = np.polyfit(np.log10(f[hf]), np.log10(p[hf]+1e-30), 1)[0]
    out['nz_orig_sfreq'] = fs
    return out

def preprocess(raw):
    raw = raw.copy()
    if abs(raw.info['sfreq'] - FS) > 1e-6:
        raw.resample(FS)
    raw.filter(1.0, 30.0, method='fir', phase='zero')
    raw.set_eeg_reference('average', projection=False)
    return raw

def epoch_and_clean(raw, max_epochs=None):
    x = raw.get_data()
    n = int(EPOCH_S * FS)
    k = x.shape[1] // n
    ep = x[:, :k*n].reshape(x.shape[0], k, n).transpose(1, 0, 2)  # (k, ch, n)
    ep = ep[2:-2] if k > 8 else ep   # drop edges (filter transients / setup)
    p2p = ep.max(-1) - ep.min(-1)              # (k, ch)
    med = np.median(p2p, axis=0)               # per channel
    bad = (p2p > 5*med).any(1) | (p2p < 0.05*med).any(1) | (p2p > 500e-6).any(1)
    ep = ep[~bad]
    if max_epochs and len(ep) > max_epochs:
        idx = np.linspace(0, len(ep)-1, max_epochs).astype(int); ep = ep[idx]
    return ep, int(bad.sum()), int(k)

def epoch_features(ep):
    """ep: (k, ch, n) -> (k, F) array, names"""
    f, p = welch(ep, fs=FS, nperseg=int(2*FS), noverlap=int(FS), window='hamming', axis=-1)  # (k,ch,f)
    tot_mask = (f >= 1) & (f < 30)
    tot = p[..., tot_mask].sum(-1)                               # (k,ch)
    bandp = {b: p[..., (f>=lo)&(f<hi)].sum(-1) for b,(lo,hi) in BANDS.items()}
    rel = {b: v/tot for b,v in bandp.items()}
    cols, names = [], []
    ci = {c:i for i,c in enumerate(CHANNELS)}
    # per-channel relative band power
    for b in BANDS:
        for c in CHANNELS:
            cols.append(rel[b][:, ci[c]]); names.append(f'rel_{b}_{c}')
    # global indices (mean over channels)
    dar = bandp['delta']/bandp['alpha']; dtabr = (bandp['delta']+bandp['theta'])/(bandp['alpha']+bandp['beta'])
    cols += [np.log10(dar).mean(1), np.log10(dtabr).mean(1)]; names += ['g_log_DAR','g_log_DTABR']
    for b in BANDS:
        cols.append(rel[b].mean(1)); names.append(f'g_rel_{b}')
    # spectral entropy (global)
    pn = p[..., tot_mask] / p[..., tot_mask].sum(-1, keepdims=True)
    se = -(pn*np.log(pn+1e-20)).sum(-1)/np.log(pn.shape[-1])
    cols.append(se.mean(1)); names.append('g_spec_entropy')
    # peak alpha frequency (posterior)
    am = (f>=7)&(f<=13); post = [ci[c] for c in ('O1','O2','P3','P4','PZ')]
    paf = f[am][np.argmax(p[:, post][..., am].mean(1), -1)]
    cols.append(paf); names.append('g_peak_alpha_freq')
    # pdBSI (van Putten 2007): mean over pairs and 1-25 Hz of |(R-L)/(R+L)|
    m = (f>=1)&(f<=25)
    L = np.stack([p[:, ci[a]][:, m] for a,_ in PAIRS], 1); R = np.stack([p[:, ci[b]][:, m] for _,b in PAIRS], 1)
    pd = np.abs((R-L)/(R+L))
    cols.append(pd.mean((1,2))); names.append('pdBSI_1_25')
    for b,(lo,hi) in BANDS.items():
        mb = (f[m]>=lo)&(f[m]<hi)
        cols.append(pd[..., mb].mean((1,2))); names.append(f'pdBSI_{b}')
    # signed asymmetry log(L/R) per pair & band  (positive = more power on LEFT)
    for a,bb in PAIRS:
        for b in BANDS:
            cols.append(np.log10(bandp[b][:, ci[a]]/bandp[b][:, ci[bb]])); names.append(f'asym_{b}_{a}_{bb}')
    # hemispheric signed asymmetry of slowing: log DAR(left) - log DAR(right)
    left = [ci[a] for a,_ in PAIRS]; right = [ci[b] for _,b in PAIRS]
    cols.append(np.log10(dar[:, left]).mean(1) - np.log10(dar[:, right]).mean(1)); names.append('hemi_logDAR_L_minus_R')
    cols.append(np.log10(bandp['delta'][:, left].sum(1)/bandp['delta'][:, right].sum(1))); names.append('hemi_delta_L_over_R')
    # Hjorth (global mean)
    d1 = np.diff(ep, axis=-1); d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = ep.var(-1), d1.var(-1), d2.var(-1)
    mob = np.sqrt(v1/v0); comp = np.sqrt(v2/v1)/mob
    cols += [mob.mean(1), comp.mean(1)]; names += ['g_hjorth_mobility','g_hjorth_complexity']
    return np.column_stack(cols), names

def process_file(path, max_epochs=None):
    raw = load_standard(path)
    nz = nuisance_features(raw)
    raw = preprocess(raw)
    ep, n_bad, n_tot = epoch_and_clean(raw, max_epochs)
    if len(ep) < 5:
        raise ValueError(f"only {len(ep)} clean epochs")
    X, names = epoch_features(ep)
    rec = dict(zip(names, np.median(X, 0)))
    rec.update(nz); rec['n_epochs'] = len(ep); rec['n_rejected'] = n_bad; rec['n_total'] = n_tot
    return rec, X, names
