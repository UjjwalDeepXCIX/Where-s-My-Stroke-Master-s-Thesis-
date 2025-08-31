# Where's My Stroke?
**Submitted to The University of Manchester for the degree of Master of Science  
in the Faculty of Science and Engineering, Department of Computer Science.**  
**Student:** Ujjwal Deep  
**Degree Program:** MSc Advanced Computer Science: AI  
**Supervisor:** Dr. Fumie Costen  

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
git clone https://github.com/YourUsername/EEG-Stroke-Classification.git
cd EEG-Stroke-Classification
pip install -r requirements.txt
````

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