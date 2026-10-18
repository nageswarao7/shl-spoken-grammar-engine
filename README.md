# Spoken Grammar Scoring Engine
### SHL AI Research Engineer Technical Assessment (Kaggle Private Challenge 2026)

**Candidate:** Nageswara Rao Vutla  
**Kaggle Username:** `nageswararao7`  
**Challenge:** [SHL Hiring Assessment 2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026)  
**Primary Metrics:** Pearson Correlation ($r$) & Root Mean Squared Error (RMSE)  
**Best Kaggle Leaderboard Score:** **`0.7394 RMSE`** (Baseline: `1.0050`)  
**Compulsory Training Data RMSE:** **`0.4158`**  
**Local Out-of-Fold Pearson Correlation ($r$):** **`0.7816`**

---

## 1. Executive Summary

This repository presents the end-to-end Machine Learning pipeline developed for the **SHL Research Engineer Spoken Grammar Scoring Assessment**. The objective is to evaluate 45–60 second spoken audio recordings and predict a continuous Mean Opinion Score (MOS) Likert grammar rating ranging from **0.0 to 5.0**.

The pipeline features:
1. **Acoustic & Prosodic Feature Extraction:** 44 core speech characteristics including 13 Mel-Frequency Cepstral Coefficients (means and standard deviations), spectral centroid, spectral rolloff, spectral bandwidth, zero-crossing rate (ZCR), RMS energy, and pause distribution.
2. **Domain-Informed Feature Engineering:** High-signal features capturing vocal tract modulation variety (`mfcc_std_sum`, linear correlation $+0.687$), pause rate, active speech energy, and spectral spread.
3. **Two-Stage Noise/Speech Segmentation:** Identification of 37 static white-noise recordings in the training data (`spec_cent > 3830 Hz`, `zcr > 0.438` $\implies$ score `0.0`), isolating them to prevent bias in human speech rating calibration.
4. **10-Fold Multimodel Stacking Ensemble:** Fusing **CatBoost**, **LightGBM**, **XGBoost**, and regularized **Ridge** regressors to produce robust, well-generalized predictions.

---

## 2. Benchmark Results & Compulsory Score

| Model Architecture | 10-Fold OOF RMSE | 10-Fold OOF Pearson $r$ | Public Leaderboard RMSE | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Mean Predictor** | 1.2390 | 0.0000 | 1.0050 | Benchmark |
| **Ridge Regressor (Acoustic Features)** | 0.8568 | 0.5379 | — | Evaluated |
| **XGBoost Regressor** | 0.7946 | 0.6211 | — | Evaluated |
| **CatBoost Regressor** | 0.7953 | 0.6225 | — | Evaluated |
| **LightGBM Regressor** | 0.8009 | 0.6133 | — | Evaluated |
| **5-Fold LightGBM + Ridge Blend** | 0.7917 | 0.7694 | 0.7619 | Submitted |
| **10-Fold Multimodel Stacking (CatBoost + LightGBM + XGBoost + Ridge)** | **0.7728** | **0.7816** | **0.7394** | **Best Submission** |

> ### Mandatory Assessment Metric
> **COMPULSORY TRAINING DATA RMSE: `0.4158`**  
> *(Explicitly computed on all 769 training samples as instructed in the competition guidelines).*

---

## 3. Key Findings & Methodology

### 3.1 Spectral Discovery: The Static-Noise Separation
During Exploratory Data Analysis (EDA), we analyzed all samples with a ground truth score of `0.0` (37 instances).
* **Static White Noise:** `spec_cent_mean` ranges from $3830\text{ Hz}$ to $3918\text{ Hz}$ ($\mu = 3870\text{ Hz}$), with `zcr_mean` $> 0.438$.
* **Human Spoken Speech:** `spec_cent_mean` ranges from $862\text{ Hz}$ to $3368\text{ Hz}$ ($\mu = 1784\text{ Hz}$), with `zcr_mean` $< 0.335$.

There is a gap of nearly $500\text{ Hz}$ between the highest human speech and lowest noise sample. Crucially, all 216 test set audio files are genuine speech samples. Training the speech models specifically on speech instances prevents the model from under-predicting valid speech scores.

### 3.2 Acoustic Correlates of Grammar & Fluency
* **Vocal Modulation (`mfcc_std_sum`):** Shows a $+0.6868$ linear correlation with grammar scores. Fluent, grammatically adept speakers constantly vary vocal tract resonances as they articulate complex syntactic phrases.
* **Hesitation & Pause Dynamics (`pause_rate`, `mean_pause_sec`):** Frequent, halting pauses correlate negatively with grammatical flow.
* **High-Frequency Rolloff (`spec_roll_mean` & `spec_bw_mean`):** Negative correlation ($-0.401$) reflecting noisy acoustics or fragmented vocal delivery.

---

## 4. Repository Structure

```
.
├── README.md                              # Technical report and documentation
├── requirements.txt                       # Python dependencies
├── shl_grammar_scoring_engine.ipynb       # Fully documented deliverable notebook
├── extract_acoustic_features.py           # Parallel acoustic & prosodic extraction
├── train_multimodel_v3.py                 # 10-fold CV training & compulsory RMSE calculation
├── submission_v3_multimodel_ensemble.csv # Final Kaggle submission file (Score: 0.7394)
├── evaluation_visualizations.png          # Residuals, actual vs predicted, feature importances
└── data/
    ├── Dataset_Final/                     # Competition dataset (train/test WAVs & CSVs)
    └── features/                          # Cached acoustic feature matrices
```

---

## 5. Quickstart & Reproducibility Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/nageswararao7/shl-spoken-grammar-engine.git
cd shl-spoken-grammar-engine

# Install required dependencies
pip install -r requirements.txt
```

### 2. Feature Extraction (Parallelized across CPU cores)
```bash
python extract_acoustic_features.py
```
*Outputs:* `data/features/train_acoustic_features.csv` (769 samples) and `data/features/test_acoustic_features.csv` (216 samples).

### 3. Model Training, Evaluation & Submission Generation
```bash
python train_multimodel_v3.py
```
*Outputs:*
* 10-Fold Cross-Validation metrics for CatBoost, LightGBM, XGBoost, and Ridge.
* Compulsory Training Data RMSE: **`0.4158`**.
* Final submission file: `submission_v3_multimodel_ensemble.csv`.

---

## 6. Interview Discussion & Technical Defense

1. **Why use acoustic/prosodic modeling?**  
   Spoken grammar assessments evaluate spontaneous speech. Grammatical breakdowns in speech are accompanied by hesitations, fragmented phonation, and reduced spectral modulation. Acoustic features capture these disfluencies directly without relying on error-prone transcripts.
2. **Why a 10-fold ensemble of diverse gradient boosting families?**  
   CatBoost utilizes symmetric oblivious decision trees that combat overfitting on dense acoustic features, LightGBM optimizes leaf-wise loss with fast histogram binning, and Ridge provides linear regularization. Blending all four models yields superior variance reduction.
3. **How was the compulsory training score computed?**  
   The ensemble was fitted on the full training partition with static noise detection applied, generating predictions for all 769 training samples to compute an unskewed RMSE of **`0.4158`**.
