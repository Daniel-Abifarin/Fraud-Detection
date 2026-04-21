# Fraud-Detection
A classification based model that predicts which transactions are fraudulent and legitimate. This is an immensely unbalanced data as well
# Credit Card Fraud Detection

## Overview
This project builds machine learning models to detect fraudulent credit 
card transactions from a highly imbalanced real-world dataset. Three 
approaches to handling class imbalance were compared — baseline, 
class weighting, and SMOTE oversampling — across two model types: 
Decision Tree and Random Forest.

## Dataset
- **Source:** ULB Machine Learning Group (Kaggle)
- **Size:** 284,807 transactions, 31 features
- **Target variable:** Class (0 = Legitimate, 1 = Fraud)
- **Class distribution:** 284,315 legitimate vs 492 fraud
- **Fraud rate:** 0.172% — severely imbalanced dataset
- **Features:** Time, Amount, and V1-V28 (PCA anonymised features)

## Project Structure
├── fraud_detection.ipynb          # Main notebook
├── creditcard.csv                 # Dataset
├── confusion_matrices.png         # Confusion matrix comparison
├── feature_importance.png         # Feature importance chart
└── README.md                      # Project documentation

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## Why This Problem is Hard
With only 0.172% fraud rate a naive model predicting legitimate 
for every transaction achieves 99.83% accuracy — making accuracy 
completely useless as a metric. The focus metrics for this project are:
- **Recall** — catching actual fraud (primary business goal)
- **F1 Score** — balancing precision and recall
- **ROC-AUC** — overall fraud ranking ability
- **Precision** — avoiding false alarms

## Methodology

### 1. Exploratory Data Analysis
- Confirmed severe class imbalance — 0.172% fraud rate
- Analysed transaction amount distributions by class:
  - Legitimate transactions range up to €25,119
  - Fraud transactions max out at €2,125 — fraudsters avoid 
    large transactions that trigger immediate alerts
- Converted Time (seconds) to Hour of Day — fraud peaks at 
  2-3am when legitimate activity is lowest and monitoring 
  is reduced. Legitimate transactions peak between 10am-11pm
- Correlation analysis revealed V17, V14, V12, V10 as strongest 
  negative predictors of fraud — confirmed by feature importance

### 2. Data Preprocessing
- Dropped one row with missing values across all columns
- Scaled Amount and Time using StandardScaler — different scale 
  from PCA-transformed V features
- Engineered Hour feature from Time column
- No encoding needed — all features already numerical (PCA transformed)

### 3. Class Imbalance Handling
Three approaches were tested and compared:

**Approach 1 — Baseline:** No imbalance handling — model trained 
on original imbalanced data

**Approach 2 — Class Weight:** `class_weight='balanced'` parameter 
tells the model to penalise fraud misclassifications more heavily 
during training. Data unchanged, only loss function adjusted.

**Approach 3 — SMOTE:** Synthetic Minority Oversampling Technique 
generates synthetic fraud samples by interpolating between existing 
fraud cases. Applied only to training data inside each CV fold 
using imblearn Pipeline to prevent data leakage.

### 4. Correct SMOTE Implementation
SMOTE was implemented using imblearn Pipeline to ensure no leakage:
```python
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])
```
This guarantees SMOTE only runs on training folds during CV and is 
automatically skipped during prediction — the gold standard approach.

### 5. Note on Hyperparameter Tuning
GridSearchCV with SMOTE pipeline was attempted but proved 
computationally prohibitive on Colab — SMOTE generating ~226,000 
synthetic samples inside each CV fold multiplied across parameter 
combinations exceeded available compute. In production this would 
run on a cloud machine (AWS/GCP/Azure) with 32+ CPU cores. Default 
Random Forest parameters were used for final comparison.

### 6. Models Built
- Decision Tree (baseline — to demonstrate overfitting and limitations)
- Random Forest — three imbalance handling approaches compared

## Results

### Decision Tree vs Random Forest
| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Decision Tree + SMOTE | 0.7755 | 0.4720 | 0.8864 |
| Random Forest Baseline | 0.8163 | 0.8743 | 0.9528 |
| Random Forest Class Weight | 0.7551 | 0.8457 | 0.9581 |
| Random Forest SMOTE | 0.8061 | 0.8103 | 0.9688 |

### Confusion Matrix Comparison — Random Forest
| | RF Baseline | RF Class Weight | RF SMOTE |
|---|---|---|---|
| Fraud Caught (TP) | 80 | 74 | 79 |
| Fraud Missed (FN) | 18 | 24 | 19 |
| False Alarms (FP) | 5 | 3 | 18 |
| Correct Legitimate (TN) | 56,859 | 56,861 | 56,846 |

## Key Findings

### Model Performance
- Random Forest dramatically outperforms Decision Tree — F1 jumped 
  from 0.47 to 0.87, ROC-AUC from 0.89 to 0.95
- RF Baseline (no imbalance handling) achieved the highest Recall 
  (0.8163) and F1 (0.8743) — catching 80 out of 98 fraud cases 
  with only 5 false alarms
- RF SMOTE achieved the highest ROC-AUC (0.9688) — best overall 
  ability to rank transactions by fraud probability
- Surprisingly imbalance handling did not dramatically improve 
  over baseline Random Forest — ensemble methods are naturally 
  more robust to imbalance than single decision trees

### Feature Importance
- V17, V14, V12, V10 are the four most important features — 
  perfectly matching correlation analysis predictions
- All top features are PCA-anonymised V features — the original 
  bank features encoded in these components are the true fraud signals
- Amount and Time showed very low feature importance despite 
  visual patterns in EDA — their relationships with fraud are 
  subtle and non-linear
- Hour of day (engineered feature) also contributed minimally 
  to the model despite the 2-3am fraud spike observed in EDA

### Business Insights
- Fraudsters make smaller transactions (max €2,125) compared to 
  legitimate customers (max €25,119) — avoiding large transaction 
  alert thresholds
- Fraud peaks between 2-3am when account monitoring is lowest 
  and owners are asleep
- RF Baseline is recommended for production — catches the most 
  fraud (80/98) with the fewest false alarms (5), minimising 
  investigation costs while maximising fraud recovery

## Model Selection Recommendation
**For maximum fraud capture:** RF Baseline — highest Recall and F1

**For fraud risk scoring/prioritisation:** RF SMOTE — highest 
ROC-AUC means best at ranking which transactions are most 
likely fraud, useful for prioritising investigation queues


## What I Would Do Next
- Run GridSearchCV on a cloud machine (AWS/GCP) with sufficient 
  compute to find optimal hyperparameters
- Try XGBoost and LightGBM — gradient boosting often outperforms 
  Random Forest on fraud detection
- Plot Precision-Recall curve and tune decision threshold below 
  0.5 to further optimise the recall/precision tradeoff
- Implement real-time scoring API using FastAPI — deploy model 
  as an endpoint that scores transactions as they arrive
- Add SHAP values for individual prediction explanations — 
  critical for regulatory compliance in banking where decisions 
  must be explainable
- Investigate the 18 missed fraud cases — what do they have in 
  common? Understanding model failures is as important as 
  measuring success

## Author
Daniel Abifarin
Electrical Engineering Student | University of Lagos
Aspiring MLOps Engineer
GitHub: github.com/Daniel-Abifarin
