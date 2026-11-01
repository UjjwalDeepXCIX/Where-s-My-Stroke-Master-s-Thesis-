# Where's My Stroke?
**Submitted to The University of Manchester for the degree of Master of Science  
in the Faculty of Science and Engineering, Department of Computer Science.**  
**Student:** Ujjwal Deep  
**Degree Program:** MSc Advanced Computer Science: AI  
**Supervisor:** Dr. Fumie Costen  

---

> ## ⚠️ Update (Nov 2026): results revised, pipeline rebuilt
> A post-submission audit found that the headline result below (**AUC 0.963**) **should not be read as stroke detection**.
> All stroke recordings come from one dataset (Figshare, China, 500 Hz, 50 Hz mains) and all normal recordings from another (TUH, US, 250/512 Hz, 60 Hz mains).
> A single non-brain feature, the 50 Hz vs 60 Hz mains-noise ratio, separates the two classes with AUC **0.9998**, so the classifier mainly learns *which site* a recording came from.
> The audit also found issues in the v1 code: train and eval features were built differently, threshold and parameters were tuned on the eval set, the GAN output was not realistic, and some preprocessing steps never reached the data.
>
> **The rebuilt pipeline is on the [`pipeline-v2`](../../tree/pipeline-v2) branch.** It uses one preprocessing path for every recording, 140 standard QEEG features, recording-level patient-grouped CV and no GAN, and it runs in about 3 min instead of about 8 h. Within the stroke cohort only (no confound):
> - Lesion side: chance level (AUC ≈ 0.46). Most lesions are subcortical.
> - Severity: EEG slowing trends with NIHSS/mRS (ρ 0.24–0.33), but nothing survives correction for multiple tests.
>
> **Next step:** stroke and control EEG from the *same* source (TUH sessions found through clinician reports vs TUH normals), with a sanity gate that nuisance-only features must score AUC ≈ 0.5.
> Details: [`pipeline_v2/README.md`](../../blob/pipeline-v2/pipeline_v2/README.md) · [audit](../../blob/pipeline-v2/docs/research-work-audit.md) · [headband validation plan](../../blob/pipeline-v2/docs/headband-validation-plan.md).
>
> The v1 code on this branch is kept unchanged as the version submitted with the dissertation.

---

## 📖 Overview
This project investigates **EEG-based stroke classification**, combining signal preprocessing, 
synthetic data generation, feature engineering, and interpretable machine learning.  
It was developed as part of the MSc dissertation in Advanced Computer Science: AI.  

---

## 📂 Project Structure
- `1_Validate_data_dir.py` — Validate dataset structure.
- `2_preprocess_and_epoch_generation.py` — EEG preprocessing and epoch generation.
- `3_generate_synthetic_eeg.py` — Conditional GAN for synthetic EEG.
- `4_extract_features.py` — Extract spectral, wavelet, entropy, and symmetry features.
- `train_model.py` — Train and evaluate XGBoost classifier with SHAP interpretability.

---

## ⚙️ Installation
Clone the repository and install requirements:

```bash
git clone https://github.com/UjjwalDeepXCIX/Where-s-My-Stroke-Master-s-Thesis-.git
cd Where-s-My-Stroke-Master-s-Thesis-
pip install -r requirements.txt
```

---

## Usage

1. **Validate Dataset**

   ```bash
   python 1_Validate_data_dir.py
   ```

2. **Preprocess and Epoch Generation**

   ```bash
   python 2_preprocess_and_epoch_generation.py
   ```

3. **(Optional) Generate Synthetic EEG**

   ```bash
   python 3_generate_synthetic_eeg.py
   ```

4. **Extract Features**

   ```bash
   python 4_extract_features.py
   ```

5. **Train and Evaluate Model**

   ```bash
   python train_model.py
   ```

---

## 📊 Key Features

* Signal harmonisation (resampling, re-referencing, QC gates)
* Power Spectral Density (Welch method)
* Brain symmetry indices (pdBSI)
* Entropy and wavelet features
* Synthetic EEG generation using Conditional GAN
* XGBoost classifier with SHAP-based interpretability

---

## 📎 Notes

* Paths for datasets and outputs should be set inside each script.
* Compatible with Python ≥3.8.
* Lightweight enough to run on CPUs; GPU recommended for GAN training.


🤝 Contributing & Collaboration

   This research is a step towards real-time, interpretable EEG-based stroke triage.
   If you are interested in:

   * improving the pipeline,

   *   testing it on new datasets,

   *   extending the feature set or models, or

   *  discussing possible clinical applications,

   please feel free to open an issue, submit a pull request, or reach out directly.
   Collaboration and feedback are warmly welcomed to strengthen this line of research.
_________