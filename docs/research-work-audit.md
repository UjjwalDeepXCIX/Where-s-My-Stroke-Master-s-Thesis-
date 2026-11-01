# Research_work folder audit (EEG stroke classification, MSc dissertation)

Reviewed 30 Oct 2026. Folder: `C:\UoM Sem-2\Research_work` (~33 GB, ~9.2k files; last real activity Sep 2025).

## What's there
- **Data**: Figshare "EEG datasets of stroke patients" (Liu & Lv, 50 subjects, 500 Hz, motor-imagery task, `participants.tsv` with NIHSS/lesion location) is the **stroke** class: 40 train (sub-01..40) / 10 eval (sub-41..50). TUH EEG is the **normal** class: 40 train / 10 eval, 250 Hz (47 files) or 512 Hz (3 files), no subject overlap between the two splits.
- **Final pipeline**: `src/github/Pipeline/` = `UjjwalDeep/Pipeline/` (identical). 1 validate → 2 preprocess/epoch → 3 cGAN synthetic → 4 features (489) → `train_model.py` (MI top-200 → RFE 50 → XGBoost, 3 param sets, SHAP, t-SNE).
- **Final run** (`src/github/src/logs/train_1755508881.log`, 18 Aug 2025): train 7,390 epochs (1,006 positive), eval 2,287 epochs (162 positive). Set 2: AUC-ROC 0.963, AUC-PR 0.754, Acc 0.966, Prec 0.828, Rec 0.654, F1 0.731 at threshold 0.30. These are the dissertation's headline numbers.
- **Write-ups**: `Paper_drafts/MSc ACS Dissertation- Ujjwal Deep- 11522913.pdf` (38 pp, 1 Sep 2025), `thesis.tex` + `references.bib`, an earlier `main.tex` (claims F1 0.819), presentation PDF/pptm + videos.
- **Older work**: `previous runs/` (CNN-LSTM .keras models Jun–Jul 2025, older XGBoost models, epoch zips), `src_previous/`, `Pipeline_previous/`, `tests/`.
- **Missing or broken**: `src/github/src/models/` is empty, so the reported Set-2 model was never saved (the save code was added after the run). `Raw research.docx` is corrupted, and `tuh_eeg.pdf` is partly corrupted (it's the TUH access form). Large duplicates: `Datasets.zip` (12 GB), `Dataset.tar`, `UjjwalDeep (2).zip`, `previous runs/epochs.zip`.

## Validity problems in the code (most to least serious)
1. **Class is completely confounded with dataset.** Every stroke recording is Figshare (500 Hz, motor-imagery task, one hospital and amplifier). Every normal recording is TUH (clinical resting EEG, US, different hardware). The model may be learning *which dataset* a recording came from rather than stroke. Supporting signs: top features include gamma (30–100 Hz) asymmetry even though data is low-passed at 40 Hz, and epoch-mean / variance features. Preprocessing also differs by class: stroke files go through an ICA path, normal files through a regression path.
2. **Train and eval features are built differently.** Train features are StandardScaled (confirmed mean 0, std 1). Eval features are not scaled, and use different Welch settings (256-sample default window vs 2 s Hamming). Channel names are also handled differently.
3. **Threshold and model choice were tuned on the eval set.** Best threshold (max F1) and best param set (max AUC) were both picked on the held-out set, so the reported numbers are optimistic. There is no grouped CV, because `groups.npy` doesn't exist.
4. **Synthetic data is unusable as built.** The GAN trained for only 50 iterations (50 batches), and discriminator loss was rising. Synthetic epochs have std ≈ 0.18, while real epochs have std ≈ 9e-6 V (~19,000× apart). No realism filtering was done. Conditioning was zeroed to class only. Synthetic data made up 624 of 7,390 training epochs (~8%), not 30%.
5. **Preprocessing doesn't do what it says.** Epochs are created *before* the 250 Hz resample and CAR, so saved epochs are 256 Hz without CAR. ICA runs with `exclude=[]`, so it removes nothing. The SNR gate uses a 45–100 Hz noise band after a 40 Hz low-pass, so it's effectively always passed. Overlap is 2 s (20%). The notch is at 60 Hz, but Figshare was recorded with 50 Hz mains (moot after the 40 Hz low-pass).
6. **Smaller issues.** In the train extractor, SampEn keeps only the last channel. The "gamma" band (30–100 Hz) is above the filter cut-off. Results are reported per epoch, not per subject (only 10 stroke eval subjects). Labels are stored as 0.9/0.1 "risk".

## Dissertation text vs. reality
- Dataset roles are reversed: the text says TUH = stroke and Figshare = healthy controls. Its own reference [17] is titled "EEG datasets of stroke patients".
- Wrong sampling rates: Figshare is described as 128 Hz but is actually 500 Hz.
- Overlap is described as 80% but is actually 20%.
- Several claims aren't in the code: 250 Hz + CAR applied, ICA eye removal, dataset-specific SNR threshold, GAN with NIHSS/location conditioning plus feature matching and gradient penalty "trained until it fooled D", synthetic filtering, 30% synthetic, early stopping with patience 10, GroupShuffleSplit, band-wise pdBSI, per-scale CWT stats.
- The ablation section describes feature-family effects that weren't actually run (the text itself says the full ablation wasn't done).
- References: [18], cited for CAR, is a free-electron-laser physics paper. [3] Heit et al. 2017 is an MRI imaging review, not a PSD+SVM study with 74% recall. The Table 1 studies (Tabibian 2018, Yu 2020, Zhang 2021, Smith 2023, Caiola 2024, Wang 2025) and their recall figures need checking against real papers.

## If this is going to be a paper
- Fix the confound first: get stroke and non-stroke EEG from the same source (e.g. TUH sessions with stroke in the clinician reports vs TUH normals), or at least run a "dataset classifier" sanity check. Report per-subject results with subject-level CV.
- Use one feature function for train and eval, and fit the scaler on train only.
- Choose threshold and params with grouped CV on train only, then evaluate once.
- Drop the GAN or rebuild it properly (normalize amplitudes, train for real, validate with PSD comparisons and a real-vs-synthetic classifier).
- Rewrite the methods section from the code and correct the references.
