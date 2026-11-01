"""Run features_v2 over every EDF and write features_recording.csv (+ per-epoch npz)."""
import os, sys, glob, json, traceback
import numpy as np, pandas as pd
from features_v2 import process_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'Datasets', 'Dataset')
OUT = os.path.join(os.path.dirname(__file__), 'out'); os.makedirs(os.path.join(OUT, 'epochs'), exist_ok=True)

meta = pd.read_csv(os.path.join(DATA, 'FIgshare', 'metadata', 'participants.tsv'), sep='\t').set_index('Participant_ID')
jobs = []
for split in ('train', 'eval'):
    for f in sorted(glob.glob(os.path.join(DATA, 'FIgshare', split, 'stroke', '*.edf'))):
        jobs.append((f, 'figshare', 1, split, os.path.basename(f).split('_')[0]))
    for f in sorted(glob.glob(os.path.join(DATA, 'TUHEEG', split, 'normal', '*.edf'))):
        jobs.append((f, 'tuh', 0, split, os.path.basename(f).split('_')[0]))
# extra TUH stroke recordings, if the user adds them later (see mine_tuh_reports.py)
for f in sorted(glob.glob(os.path.join(ROOT, 'Datasets', 'TUH_stroke', '**', '*.edf'), recursive=True)):
    jobs.append((f, 'tuh', 1, 'tuh_stroke', os.path.basename(f).split('_')[0]))

csv = os.path.join(OUT, 'features_recording.csv')
done = set(pd.read_csv(csv)['file']) if os.path.exists(csv) else set()
rows = []
for i, (f, src, y, split, subj) in enumerate(jobs):
    base = os.path.basename(f)
    if base in done: continue
    try:
        rec, X, names = process_file(f, max_epochs=150)
        rec.update(file=base, source=src, label=y, split=split, subject=subj)
        if src == 'figshare' and subj in meta.index:
            for k in ('Age','Gender','ParalysisSide','StrokeLocation','NIHSS','mRS','Duration'):
                rec['meta_'+k] = meta.loc[subj, k]
        np.savez_compressed(os.path.join(OUT, 'epochs', base.replace('.edf', '.npz')), X=X.astype(np.float32), names=np.array(names))
        pd.DataFrame([rec]).to_csv(csv, mode='a', header=not os.path.exists(csv), index=False)
        print(f"[{i+1}/{len(jobs)}] OK {base} epochs={rec['n_epochs']} rej={rec['n_rejected']}", flush=True)
    except Exception as e:
        print(f"[{i+1}/{len(jobs)}] FAIL {base}: {e}", flush=True)
print("DONE", flush=True)
