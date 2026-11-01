"""
pipeline_v2 / experiments.py
E1  Confound test: can we tell stroke(Figshare) from normal(TUH) with features that carry NO stroke physiology?
E2  Lesion lateralisation inside the stroke cohort (same hospital/device/task -> no site confound).
E3  Severity: QEEG indices vs NIHSS / mRS inside the stroke cohort.
All CV is subject-grouped; scaler/selection live inside the CV pipeline; nothing is tuned on a test fold.
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(__file__), 'out')
d = pd.read_csv(os.path.join(OUT, 'features_recording.csv'))
R = {}

PHYS = [c for c in d.columns if c.startswith(('rel_','g_','pdBSI','asym_','hemi_'))]
NUIS = ['nz_line50_vs_60_log','nz_abs_power_log','nz_hf_slope']

def logreg(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
def xgb(): return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=2, verbosity=0)

def grouped_auc(X, y, g, model_fn, reps=10):
    aucs, bacc = [], []
    for r in range(reps):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
        p = np.zeros(len(y))
        for tr, te in cv.split(X, y, g):
            m = model_fn(); m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y, p)); bacc.append(balanced_accuracy_score(y, p>=0.5))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(bacc))

# ---------------- E1 ----------------
y = d['label'].values; g = d['subject'].values
e1 = {}
for name, cols in [('physiology_features', PHYS), ('nuisance_only', NUIS),
                   ('line_noise_only', ['nz_line50_vs_60_log']), ('orig_sampling_rate_only', ['nz_orig_sfreq'])]:
    X = d[cols].values
    for mname, mf in [('logreg', logreg), ('xgb', xgb)]:
        e1[f'{name}|{mname}'] = grouped_auc(X, y, g, mf, reps=5)
# held-out split, trained once, no tuning on eval (v1 protocol done correctly)
tr, te = d['split'].eq('train').values, d['split'].eq('eval').values
for name, cols in [('physiology_features', PHYS), ('nuisance_only', NUIS)]:
    m = xgb().fit(d.loc[tr, cols].values, y[tr]); p = m.predict_proba(d.loc[te, cols].values)[:,1]
    e1[f'heldout|{name}|xgb'] = float(roc_auc_score(y[te], p))
R['E1_confound'] = e1

# ---------------- E2 ----------------
s = d[d['source'].eq('figshare')].copy()
s['lesion_left'] = s['meta_ParalysisSide'].str.lower().eq('right').astype(int)   # right-sided paralysis -> left-hemisphere lesion
LAT = [c for c in PHYS if c.startswith(('asym_','hemi_'))]
X2, y2 = s[LAT].values, s['lesion_left'].values
def cv_auc_plain(X, y, mf, reps=20, perm=False, seed=0):
    out = []
    for r in range(reps):
        yy = np.random.default_rng(seed*1000+r).permutation(y) if perm else y
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=r)
        p = np.zeros(len(yy))
        for tr_, te_ in cv.split(X, yy):
            m = mf(); m.fit(X[tr_], yy[tr_]); p[te_] = m.predict_proba(X[te_])[:,1]
        out.append(roc_auc_score(yy, p))
    return np.array(out)
e2 = {'n_left_lesion': int(y2.sum()), 'n_right_lesion': int((1-y2).sum())}
for mname, mf in [('logreg', logreg), ('xgb', xgb)]:
    real = cv_auc_plain(X2, y2, mf, reps=20)
    null = np.array([cv_auc_plain(X2, y2, mf, reps=1, perm=True, seed=k)[0] for k in range(200 if mname=='logreg' else 100)])
    e2[mname] = {'auc_mean': float(real.mean()), 'auc_sd': float(real.std()),
                 'perm_p': float((np.sum(null >= real.mean()) + 1) / (len(null) + 1)), 'null_mean': float(null.mean())}
# single, pre-specified physiological marker: hemispheric delta asymmetry (no fitting at all)
for f in ['hemi_delta_L_over_R', 'hemi_logDAR_L_minus_R']:
    a, b = s.loc[s.lesion_left==1, f], s.loc[s.lesion_left==0, f]
    e2[f] = {'auc_no_training': float(roc_auc_score(y2, s[f])), 'mannwhitney_p': float(mannwhitneyu(a, b).pvalue),
             'median_left_lesion': float(a.median()), 'median_right_lesion': float(b.median())}
R['E2_lateralisation'] = e2

# ---------------- E3 ----------------
e3 = {}
for f in ['g_log_DAR','g_log_DTABR','pdBSI_1_25','pdBSI_delta','g_rel_delta','g_rel_alpha','g_peak_alpha_freq','g_spec_entropy']:
    for t in ['meta_NIHSS','meta_mRS']:
        rho, p = spearmanr(s[f], s[t]); e3[f'{f}~{t}'] = {'rho': round(float(rho),3), 'p': round(float(p),4)}
R['E3_severity'] = e3

json.dump(R, open(os.path.join(OUT, 'results.json'), 'w'), indent=2)
print(json.dumps(R, indent=1))
