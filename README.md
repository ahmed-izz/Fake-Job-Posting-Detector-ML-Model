# Fake Job Posting Detector (NLP + Machine Learning)

A simple college project that predicts whether a job posting is **Likely Legitimate (0)** or **Potentially Fraudulent (1)** using **Natural Language Processing (NLP)** on the job text.

The project includes:
- A training script that builds a model using the Kaggle dataset
- A simple **web interface** (Streamlit) where you paste a **job URL** (or job text) and get a prediction

---

**weblink:** https://fake-job-posting-detector-ml-model-ndqqexdfkscbadbbqqjxxn.streamlit.app/

---

## 1) Problem Statement

Online job boards sometimes contain fraudulent job postings designed to scam job seekers or collect sensitive information.

**Goal:** Build a machine learning model that flags potentially fraudulent job posts using:
- Job posting text (title, description, requirements, etc.)
- A decision threshold tuned to prioritize **high precision** (avoid falsely accusing legitimate job posts)

---

## 2) Dataset

Source: Kaggle — *Real or Fake Job Postings Prediction* (University of the Aegean)

- ~18,000 job postings
- Target column: `fraudulent`  
  - `0` = Legitimate  
  - `1` = Fraudulent  
- Imbalanced classes (~5% fraud)

You must download the dataset CSV separately and update the path inside `train_model.py`.

---

## 3) Approach (Simple + Understandable)

### Model
- **TF‑IDF Vectorizer** (turns text into numeric features)
- **Logistic Regression** (binary classifier)

### Why this model?
- Works well for text classification
- Fast to train
- Easy to explain for a college project

### Class Imbalance Handling
- Uses `class_weight="balanced"` to reduce bias toward the majority class (legitimate posts)

### Metrics
Because fraud is rare, accuracy alone is misleading. The project focuses on:
- Precision / Recall / F1-score (especially for fraud class)
- ROC-AUC
- PR-AUC (Average Precision)

### Threshold Tuning (High Precision)
Instead of always using the default `0.5` threshold, we automatically choose a threshold that aims for **precision ≥ 0.90** on the test set.

This matches the requirement:
> Avoid flagging legitimate jobs as fraud (high precision).

---

## 4) Project Files

- `train_model.py`  
  Trains the model, evaluates it, selects a high-precision threshold, and saves it.

- `fake_job_model.joblib`  
  Saved artifact containing:
  - trained pipeline (TF‑IDF + Logistic Regression)
  - chosen decision threshold

- `app.py`  
  Streamlit web app:
  - Paste a job URL → extracts visible page text → predicts fraud/legit
  - If URL extraction fails, you can paste the job description manually

---

## 5) Setup Instructions (Windows)

### 5.1 Create and activate a virtual environment
In your project folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
