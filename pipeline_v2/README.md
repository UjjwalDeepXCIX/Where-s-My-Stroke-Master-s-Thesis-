# pipeline_v2: rebuilt EEG stroke pipeline (Oct 2026)

Replaces `src/github/Pipeline` (v1). How to run:
```
pip install mne xgboost scikit-learn scipy pandas
python extract_all.py      # ~3 min for all 100 recordings -> out/features_recording.csv
python experiments.py      # -> out/results.json
```

## What changed from v1, and why
| v1 problem | v2 fix |
|---|---|
| Resample + CAR applied after epoching, so they never reached the data | resample to 250 Hz, filter 1–30 Hz, CAR on the continuous signal, then epoch |
| Train and eval features built by different code (eval not scaled, different Welch settings) | one `process_file()` for every recording; scaling happens inside each CV fold |
| Threshold and params picked on the eval set | grouped CV only; nothing is tuned on test folds |
| Epoch-level metrics (same subject's epochs counted as independent samples) | one row per recording (median over clean epochs), patient-grouped CV (TUH patients with 2–3 sessions stay in one fold) |
| GAN output ~19,000× real amplitude, trained 50 steps | GAN removed |
| 489 features (gamma above the filter cut-off, per-channel CWT stats) | 140 literature QEEG features: relative band power, DAR, DTABR, pdBSI (van Putten), signed L/R asymmetry, spectral entropy, peak alpha, Hjorth |
| ~8 h runtime | ~3 min |

## Results (out/results.json)
**E1: the stroke-vs-normal task with Figshare-vs-TUH is a site-detection task** (5-fold patient-grouped CV, 5 repeats; AUC):
- Physiology features: 0.96 (logreg), 0.94 (XGB); held-out split 0.98
- 50 Hz vs 60 Hz mains-noise ratio alone: **0.9998**
- Original sampling rate alone: 0.94–0.95
Conclusion: any score on this pairing is uninterpretable, because a single feature with no brain content separates the classes perfectly.

**E2: lesion side within the 50 stroke patients** (same hospital/device, so no confound): AUC 0.45–0.47, permutation p ≈ 0.6, i.e. chance.
Most lesions are subcortical (pons, basal ganglia, paraventricular), which scalp EEG barely sees. In the 10 cortical-lesion patients, delta asymmetry points the right way (C3–C4 0.71, T3–T4 0.81, P3–P4 0.86 AUC, no training), but n = 10 (3 vs 7) is too small to claim anything.

**E3: severity within the stroke cohort** (Spearman, n = 50): slowing goes with severity, as the literature expects, but it's weak.
log DAR vs NIHSS ρ = 0.24 (p = 0.10); DTABR vs NIHSS ρ = 0.27 (p = 0.06); rel. delta vs mRS ρ = 0.28 (p = 0.048); spectral entropy vs mRS ρ = −0.33 (p = 0.02). None survive correction for 16 tests.

## The way forward
Build stroke and control cohorts from the same source: TUH stroke sessions found through clinician reports (`mine_tuh_reports.py`) vs the TUH normals already here. Then re-run E1 and check that the nuisance-only AUC drops to about 0.5. That is the sanity gate any stroke-vs-control result has to pass.

## Data layout
Data is not included (TUH EEG is under a data-use agreement; Figshare data is at its original source). The scripts expect a `Datasets/` folder next to `pipeline_v2/`:
```
Datasets/Dataset/FIgshare/{train,eval}/stroke/*.edf
Datasets/Dataset/FIgshare/metadata/participants.tsv
Datasets/Dataset/TUHEEG/{train,eval}/normal/*.edf
Datasets/TUH_stroke/**/*.edf        # optional, from mine_tuh_reports.py
```
Logic tests on synthetic signals (no data needed): `python tests/test_logic.py`.
See also `docs/research-work-audit.md` (review of v1 and the dissertation) and `docs/headband-validation-plan.md`.
