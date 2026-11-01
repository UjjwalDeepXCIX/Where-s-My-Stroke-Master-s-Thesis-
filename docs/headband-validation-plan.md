# Headband MVP: logic validation plan (1 Nov 2026)

Goal: prove the EEG processing logic works as intended before building the headband MVP. Each stage has a pass/fail gate, and we don't move to the next stage until the gate is passed. Stroke detection is deferred: it needs same-source TUH stroke data (see `pipeline_v2/README.md`).

## Stage 0: synthetic ground truth (done, `pipeline_v2/tests/test_logic.py`)
12/13 checks pass. The pipeline recovers injected alpha (rel 0.88), peak alpha frequency (10 and 9 Hz exact), DAR rise when alpha is removed, and a left-sided delta "lesion" (hemi delta L/R -0.01 to +1.63, pdBSI delta 0.32 to 0.65). The 50/60 Hz mains fingerprint has the correct sign. Physiology features are unaffected by mains frequency, amplifier gain (x3) and sampling rate (256 vs 500 Hz), with diff 0.000 in each case. Blink epochs are rejected.

Open item: the absolute 500 µV rejection drops every epoch if amplitudes are 10x (or in the wrong units). `process_file` then raises "only 0 clean epochs". The headband loader must convert µV to V and check units explicitly, because headband SDKs (e.g. Muse) usually output µV.

## Changes needed for a 4-channel headband (Muse-style TP9, AF7, AF8, TP10; Fpz reference)
- No CAR: an average reference over 4 channels is meaningless. Use the device reference or a bipolar montage instead.
- Peak alpha frequency currently uses posterior channels, which a headband doesn't have. Use the TP9/TP10 channels instead.
- Asymmetry reduces to 2 pairs (AF7-AF8 and TP9-TP10). Forehead channels are dominated by blinks and EMG, so artefact handling matters more.

## Stage 1: known-physiology tests on public data, headband-proxy channels only
| Test | Dataset | Proxy channels | Gate |
|---|---|---|---|
| Eyes closed vs open (alpha) | PhysioNet EEGMMIDB runs R01/R02, 109 subjects | Fp1, Fp2, T7, T8 | subject-grouped AUC ≥ 0.90 |
| Sleep staging | PhysioNet Sleep-EDF Expanded (sleep-cassette) | Fpz-Cz (single channel) | Cohen's kappa ≥ 0.70, subject-wise CV |
| Mental workload vs rest | PhysioNet "EEG During Mental Arithmetic Tasks", 36 subjects | Fp1, Fp2, T3, T4 | AUC ≥ 0.75 |
| Site/nuisance control | each dataset | nuisance features only | AUC ≈ 0.5 within dataset |

## Stage 2: robustness
Cross-dataset transfer, channel dropout, simulated dry-electrode noise and motion artefacts.

## Stage 3: MVP
Off-the-shelf 4-channel headband streamed via BrainFlow or muselsl, running the same feature code live. First check: own eyes-open vs eyes-closed recording. Wellness features only (sleep, stress/relaxation, focus). Recommendations are driven by trends and goals, with no diagnostic claims.
