"""Stage-0 logic checks: synthetic EEG with KNOWN ground truth -> does pipeline_v2 recover it?
Run: python tests/test_logic.py   (from pipeline_v2/)"""
import sys, os, numpy as np, mne
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features_v2 as F
mne.set_log_level("error")
RNG = np.random.default_rng(0)
CH = F.CHANNELS; LEFT = {'FP1','F7','F3','T3','C3','T5','P3','O1'}

def pink(n, fs, rng):
    w = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1/fs); f[0] = 1
    return np.fft.irfft(w/np.sqrt(f), n)

def make(fs=500, dur=120, alpha_hz=10.0, alpha_amp=20e-6, left_delta=0.0, line=None, gain=1.0, blinks=0, seed=0):
    RNG = np.random.default_rng(seed)
    n = int(fs*dur); t = np.arange(n)/fs; X = np.zeros((19, n))
    for i, c in enumerate(CH):
        x = pink(n, fs, RNG); x = x/x.std()*10e-6
        x += alpha_amp*np.sin(2*np.pi*alpha_hz*t + RNG.uniform(0, 2*np.pi))
        if c in LEFT and left_delta: x += left_delta*np.sin(2*np.pi*2.0*t + RNG.uniform(0, 2*np.pi))
        if line: x += 30e-6*np.sin(2*np.pi*line*t)
        X[i] = x
    for _ in range(blinks):                        # big frontal deflections
        s = RNG.integers(fs*10, n-fs*10); w = np.hanning(int(0.4*fs))
        for c in ('FP1','FP2'): X[CH.index(c), s:s+len(w)] += 400e-6*w
    raw = mne.io.RawArray(X*gain, mne.create_info(CH, fs, 'eeg'))
    return raw

def feats(raw):
    nz = F.nuisance_features(raw); r = F.preprocess(raw)
    ep, nbad, ntot = F.epoch_and_clean(r); X, names = F.epoch_features(ep)
    d = dict(zip(names, np.median(X, 0))); d.update(nz); d['n_bad'] = nbad; d['n_tot'] = ntot; return d

results = []
def check(name, cond, detail):
    results.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

base = feats(make())
check("alpha dominates when alpha injected", base['g_rel_alpha'] > 0.5, f"g_rel_alpha={base['g_rel_alpha']:.2f}")
check("peak alpha freq = 10 Hz", abs(base['g_peak_alpha_freq']-10) <= 0.5, f"PAF={base['g_peak_alpha_freq']:.2f}")
d9 = feats(make(alpha_hz=9.0))
check("peak alpha freq tracks 9 Hz", abs(d9['g_peak_alpha_freq']-9) <= 0.5, f"PAF={d9['g_peak_alpha_freq']:.2f}")
na = feats(make(alpha_amp=0))
check("no alpha -> DAR rises", na['g_log_DAR'] > base['g_log_DAR'] + 0.5, f"logDAR {base['g_log_DAR']:.2f} -> {na['g_log_DAR']:.2f}")
ld = feats(make(left_delta=40e-6))
check("left delta -> hemi_delta_L_over_R > 0", ld['hemi_delta_L_over_R'] > 0.3 and abs(base['hemi_delta_L_over_R']) < 0.15,
      f"base {base['hemi_delta_L_over_R']:+.2f}, left-lesion {ld['hemi_delta_L_over_R']:+.2f}")
check("left delta -> asym_delta_T3_T4 > 0", ld['asym_delta_T3_T4'] > 0.3, f"{ld['asym_delta_T3_T4']:+.2f}")
check("left delta -> pdBSI_delta rises", ld['pdBSI_delta'] > base['pdBSI_delta'] + 0.1, f"{base['pdBSI_delta']:.2f} -> {ld['pdBSI_delta']:.2f}")
m50, m60 = feats(make(line=50)), feats(make(line=60))
check("mains fingerprint sign 50 vs 60", m50['nz_line50_vs_60_log'] > 1 and m60['nz_line50_vs_60_log'] < -1,
      f"50Hz site {m50['nz_line50_vs_60_log']:+.1f}, 60Hz site {m60['nz_line50_vs_60_log']:+.1f}")
phys = [k for k in base if k.startswith(('rel_','g_','pdBSI','asym_','hemi_')) and k!='g_peak_alpha_freq']
dev = max(abs(m50[k]-m60[k]) for k in phys)
check("physiology features blind to mains frequency", dev < 0.05, f"max |diff| = {dev:.3f}")
g = feats(make(gain=3.0)); g0 = feats(make(gain=1.0))
dev = max(abs(g[k]-g0[k]) for k in phys if k.startswith(('rel_','g_rel','asym','pdBSI')))
check("relative features gain-invariant (x3 amplifier gain)", dev < 0.05, f"max |diff| = {dev:.3f}")
try:
    gb = feats(make(gain=10.0)); ok = gb['n_bad'] < 0.2*gb['n_tot']; det = f"rejected {gb['n_bad']}/{gb['n_tot']}"
except Exception as e:
    ok, det = False, f"crashed: all epochs rejected ({type(e).__name__})"
check("x10 gain does not wipe out epochs (absolute 500uV reject)", ok, det)
from scipy.signal import resample_poly
src = make(fs=4000); Xs = src.get_data()
mk = lambda fs, up, dn: mne.io.RawArray(resample_poly(Xs, up, dn, axis=1), mne.create_info(CH, fs, 'eeg'))
r256 = feats(mk(256, 8, 125)); r500 = feats(mk(500, 1, 8))
dev = max(abs(r256[k]-r500[k]) for k in phys if k.startswith(('rel_','g_rel')))
check("sampling-rate invariant (256 vs 500 Hz)", dev < 0.05, f"max |diff| = {dev:.3f}")
bl = feats(make(blinks=8))
check("blink epochs rejected", bl['n_bad'] >= 6, f"rejected {bl['n_bad']}/{bl['n_tot']} (8 blinks injected)")
print(f"\n{sum(p for _,p in results)}/{len(results)} checks passed")
