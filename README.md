# Machine Learning Projects — Emmanuel Nonso Elege

Four end-to-end machine learning projects spanning time-series deep learning, NLP, and classical supervised modelling. Each covers the full pipeline: data ingestion, exploratory analysis, preprocessing, model training, evaluation, and interpretation.

MSc Data Science (Distinction), Ulster University.

**Flagship work:** scratch detection from wearable IMU sensors using Transformer architectures — presented at ISSC 2026. Full write-up in its own repository: [eczema-scratch-detection](https://github.com/Nuel-elege/eczema-scratch-detection)

## Overview

| Project | Domain | Data | Key methods | Headline result |
|---|---|---|---|---|
| Eczema Scratch Detection | Time-series / health AI | Custom IMU dataset, 12 channels @ 20 Hz | Conformer, Transformer, Random Forest, SVM | Zero false positives on non-movement |
| Political Bias Detection | NLP / deep learning | 3,458 news articles | LSTM, TF-IDF, RandomOverSampler | 88% test accuracy, 5-class |
| Loan Approval Prediction | Risk modelling | 45,000 applications | Random Forest, KNN, Logistic Regression | 91.8% accuracy, 10-fold CV |
| Customer Churn Prediction | Applied ML / BI | 7,043 customer records | Random Forest, K-Means, SMOTE | 3 actionable customer segments |

## Eczema Scratch Detection via Wearable IMU Sensors

**→ [Full repository and write-up](https://github.com/Nuel-elege/eczema-scratch-detection) · Presented at ISSC 2026**

Automatic detection of scratching behaviour from wrist-worn inertial sensors, as an objective alternative to patient self-report in eczema severity assessment.

- Custom dataset collected for this project — accelerometer, gyroscope, total acceleration and magnetometer, 12 channels at 20 Hz. Not a public benchmark.
- Sliding-window preprocessing (size 40, step 20) with majority-label voting for binary sequence classification.
- Compared a Standard Transformer (TransformerEncoder, 2 layers, 4 heads, d_ff = 512) against a Conformer (Conv1D + TransformerEncoder), with Random Forest and SVM baselines.
- Class-weighted cross-entropy (10x on the scratch class), Adam, early stopping at patience 5.
- Evaluated by scratch-event counting rather than per-frame accuracy — consecutive positive blocks collapsed into discrete events.
- Zero false positives on flat and non-movement scenarios; 20 scratch events correctly detected on validation; 4–6 false events under deliberately confusable touch and slow-motion conditions.
- Sensitivity analysis: grid search over window {20, 40, 60, 80} x step {10, 20, 30, 40}, F1 visualised as heatmaps.

`Python` `PyTorch` `scikit-learn` `IMU` `time-series` `Transformer` `Conformer`

## Political Bias Detection in News Articles

Multi-class NLP classifier predicting the political leaning of news articles across five categories: Left, Lean Left, Centre, Lean Right, Right.

- 3,458 articles from multiple sources; EDA covering bias distribution, source-level breakdowns and per-category word clouds.
- Text preprocessing: URL and special-character removal, lowercasing, stop-word analysis. RandomOverSampler for label imbalance.
- TF-IDF vectorisation (5,000 features) plus Keras tokenisation with sequence padding (maxlen 300).
- LSTM: Embedding(5,000 → 128) → LSTM(100, dropout 0.3) → Dense(128, ReLU) → Dropout(0.3) → softmax.
- 88% test accuracy after 10 epochs. Confusion matrix shows stronger separation on clearly partisan classes than on the "Lean" categories — the harder, more interesting failure mode.

`Python` `PyTorch` `LSTM` `TF-IDF` `NLP`

[View Code](https://github.com/Nuel-elege/data-science-projects/tree/main/Political-Bias-Detection)

## Loan Approval Prediction

Binary classification of loan approval across 45,000 applications covering applicant demographics, financial profile and loan characteristics.

- Outlier handling: median replacement for implausible age (>100) and employment experience (>80).
- Encoding: binary (gender, prior defaults), ordinal (education), one-hot (home ownership, loan intent).
- Log transformation on 7 skewed features; RobustScaler for outlier resistance; SMOTE for a ~78/22 class split.
- Correlation-based feature selection (|r| > 0.1 with target; de-duplicated pairs above |r| > 0.8).
- Compared Logistic Regression, Random Forest and KNN (k=3) via confusion matrices, 10-fold cross-validation, ROC-AUC and feature importances.
- Random Forest: ~91.8% accuracy, low variance across folds. Top predictors: loan-to-income ratio, interest rate, prior defaults.

`Python` `scikit-learn` `Random Forest` `KNN` `SMOTE`

[View Code](https://github.com/Nuel-elege/data-science-projects/tree/main/Loan-Approval-Prediction)

## Customer Churn Prediction in Telecom

Churn modelling and customer segmentation over 7,043 telecom records, combining prediction with actionable business intelligence.

- Median imputation for missing TotalCharges; label encoding; SMOTE for class imbalance.
- EDA across 18 features via KDE, bar and count plots; heatmap-based feature selection (correlation threshold 0.80).
- Logistic Regression, Random Forest (n=100) and Decision Tree, evaluated on accuracy, confusion matrix and ROC-AUC.
- K-Means segmentation (k=3, elbow method): Loyal Long-Term, Price-Sensitive, High-Risk Short-Term.
- Highest-risk profile: tenure under 20 months, month-to-month contract, monthly charges above $60, electronic check payment. Retention strategies derived from the segments.

`Python` `scikit-learn` `pandas` `K-Means` `SMOTE`

[View Code](https://github.com/Nuel-elege/data-science-projects/tree/main/Customer-Churn-Prediction)

## Repository structure

```
data-science-projects/
├── Customer-Churn-Prediction/
├── Political-Bias-Detection/
├── Loan-Approval-Prediction/
└── README.md
```

The eczema scratch detection work previously lived here as a subfolder; it now has its own repository: [eczema-scratch-detection](https://github.com/Nuel-elege/eczema-scratch-detection).

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
pip install torch tensorflow
pip install wordcloud nltk plotly
```

Emmanuel Nonso Elege — nuel.elege@gmail.com · [LinkedIn](https://www.linkedin.com/in/emmanuel-elege) · [Portfolio](https://nuel-elege.github.io/portfolio/)

Emmanuel Nonso Elege — nuel.elege@gmail.com · [LinkedIn](https://www.linkedin.com/in/emmanuel-elege) · [Portfolio](https://nuel-elege.github.io/portfolio/)
