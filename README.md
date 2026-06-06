# Data Science Projects — Emmanuel Nonso Elege

> University of Ulster, Magee Campus · School of Computing, Engineering and Intelligence Systems
>
> **Repository Visibility: Private** 🔒

This repository contains four data science and machine learning coursework projects completed as part of the MSc Data Science programme. Each project follows a full pipeline from data ingestion and exploratory analysis through to model training, evaluation, and insight delivery.

---

## Projects Overview

| Module | Project Title | Dataset | Key Algorithms |
|--------|--------------|---------|----------------|
| COM735 | Churn Analysis — Optimizing Customer Retention in Telecom | customer_churn_telecom_services.csv (~7,043 records) | Logistic Regression, Random Forest, Decision Tree, K-Means |
| COM737 | Political Bias Detection in News Articles | Political_Bias.csv (~3,458 articles) | LSTM, TF-IDF, RandomOverSampler |
| COM740 | Loan Approval Prediction | loan_data.csv (~45,000 records) | Logistic Regression, Random Forest, KNN |
| Dissertation | Detecting Eczema Symptoms via Scratch Detection | COMBINED.csv (IMU wearable sensor data) | Random Forest, SVM, Conformer Transformer, Standard Transformer |

---

## COM735 — Churn Analysis: Optimizing Customer Retention Strategies in Telecom

**File:** `COM735-Churn-Analysis/churn_analysis.py`

### Overview
This project investigates customer churn in the telecommunications sector using Business Intelligence (BI) and machine learning. Over 7,000 customer records are analysed to identify key churn drivers and derive actionable retention strategies.

### Dataset
- **Source:** Telecom customer dataset (~7,043 records)
- **Features:** Demographics, service usage, financial data (MonthlyCharges, TotalCharges), contract type, payment method
- **Target:** Binary `Churn` (Yes/No)

### Methodology
1. **Data Preprocessing** — Median imputation for missing TotalCharges; label encoding for categorical features; SMOTE to handle class imbalance
2. **Exploratory Data Analysis (EDA)** — KDE plots, bar charts, and count plots across 18 features to identify churn patterns
3. **Correlation Analysis** — Heatmap-based feature selection; removal of highly correlated features (threshold = 0.80)
4. **Predictive Modelling** — Logistic Regression, Random Forest (n=100), Decision Tree; evaluated via Accuracy, Confusion Matrix, and ROC-AUC curves
5. **Customer Segmentation** — K-Means clustering (k=3, Elbow Method) into: *Loyal Long-Term*, *Price-Sensitive*, and *High-Risk Short-Term* customers

### Key Findings
- Customers with tenure < 20 months, month-to-month contracts, monthly charges > $60, and electronic check payments are the highest-risk churn group
- **Random Forest** delivered the best predictive accuracy
- Retention strategies derived: contract incentives, autopay rewards, value-added bundles, and loyalty programmes

---

## COM737 — Political Bias Detection in News Articles (NLP & Deep Learning)

**File:** `COM737-Sentiment-Analysis/political_bias_detection.py`

### Overview
A multi-class NLP classification project that detects the political leaning (Left, Lean Left, Centre, Lean Right, Right) of news articles using deep learning. The project addresses political media bias as a growing societal concern by building an automated classifier trained on real-world news data.

### Dataset
- **Source:** Political_Bias.csv (~3,458 news articles from multiple sources)
- **Features:** Article title, full text, source, bias label
- **Target:** 5-class political bias label (encoded numerically)

### Methodology
1. **EDA** — Bias distribution plots, source-level bias breakdowns, and word clouds per bias category
2. **Text Preprocessing** — URL/special character removal, lowercasing, stop-word analysis
3. **Class Balancing** — `RandomOverSampler` applied to address label imbalance
4. **Feature Engineering** — TF-IDF vectorisation (max 5,000 features) + Keras Tokenizer with sequence padding (maxlen=300)
5. **Deep Learning Model** — LSTM architecture: Embedding(5000→128) → LSTM(100, dropout=0.3) → Dense(128, ReLU) → Dropout(0.3) → Softmax output
6. **Evaluation** — Training/validation accuracy & loss curves, confusion matrix, classification report

### Key Results
- Achieved **~88% test accuracy** after 10 training epochs
- Confusion matrix reveals stronger performance on clearly partisan categories (Left, Right) vs. nuanced ones (Lean Left, Lean Right)

---

## COM740 — Loan Approval Prediction

**File:** `COM740-Loan-Approval/loan_approval.py`

### Overview
This project builds a machine learning pipeline to predict whether a loan application will be approved or rejected. The dataset of 45,000 applications covers applicant demographics, financial profile, and loan characteristics.

### Dataset
- **Source:** loan_data.csv (~45,000 records)
- **Features:** Age, gender, education, income, employment experience, home ownership, loan amount & intent, interest rate, credit score, previous defaults
- **Target:** Binary `loan_status` (0 = Rejected, 1 = Approved)

### Methodology
1. **Outlier Handling** — Median replacement for extreme age (>100) and employment experience (>80)
2. **Encoding** — Binary encoding (gender, defaults), ordinal encoding (education), one-hot encoding (home ownership, loan intent)
3. **Log Transformation** — Applied to 7 skewed features to reduce distributional skewness
4. **Feature Selection** — Correlation-based: features with |r| > 0.1 with loan_status selected; highly correlated pairs (|r| > 0.8) de-duplicated
5. **Scaling** — `RobustScaler` (robust to outliers); SMOTE for class imbalance (original ~78%/22% split)
6. **Models** — Logistic Regression, Random Forest, K-Nearest Neighbours (k=3)
7. **Evaluation** — Confusion matrices, 10-fold cross-validation, ROC-AUC curves, feature importance plots

### Key Results
- **Random Forest** achieved the best performance with ~**91.8% accuracy**
- Key predictors: loan_percent_income, loan_int_rate, previous_loan_defaults_on_file
- Cross-validation confirms generalisation with low variance across folds

---

## Dissertation — Detecting Eczema Symptoms Using Classification Algorithms

**File:** `Dissertation-Eczema-Detection/eczema_scratch_detection.py`

### Overview
This dissertation investigates the automatic detection of scratching behaviour — a primary symptom of eczema — using wearable Inertial Measurement Unit (IMU) sensor data. The project compares traditional machine learning classifiers against novel deep learning architectures (Transformer-based) for binary sequence classification.

### Dataset
- **Source:** Custom collected wearable sensor data (COMBINED.csv for training; VALID, FLAT, NONE, TOUCH, SLOW CSV files for evaluation)
- **Features:** 12 IMU channels — accelerometer (x,y,z), gyroscope (x,y,z), totalAcceleration (x,y,z), magnetometer (x,y,z)
- **Target:** Binary `scratch_x` (1 = Scratching, 0 = Non-Scratching)

### Methodology
1. **Classical Models** — Random Forest and SVM trained on raw frame-level features
2. **Sliding Window Preprocessing** — Overlapping windows (size=40, step=20) applied to raw time-series; majority-label voting assigns window class
3. **Transformer Architectures:**
   - **Standard Transformer** — TransformerEncoder (2 layers, 4 heads, d_ff=512) with final time-step classification
   - **Conformer** — Conv1D + TransformerEncoder (captures local and global temporal dependencies)
4. **Training Strategy** — Adam optimiser, class-weighted cross-entropy loss (10× weight on scratch class), early stopping (patience=5)
5. **Evaluation** — Confusion matrices, F1 scores, scratch event counting (consecutive prediction blocks)
6. **Sensitivity Analysis** — Grid search over window sizes {20, 40, 60, 80} × step sizes {10, 20, 30, 40} with heatmap visualisation

### Key Results
- Both Transformer models successfully detect scratching events in the VALID dataset (20 scratch actions predicted)
- **Flat/None** scenarios produce zero false positives, demonstrating strong specificity
- Touch and slow-motion scenarios produce minimal false predictions (4–6 events)
- Window size and step size significantly impact F1 scores, visualised via heatmap grids
- Random Forest achieves strong baseline performance with interpretable feature importances

---

## Repository Structure

```
data-science-projects/
├── COM735-Churn-Analysis/
│   └── churn_analysis.py          # EDA, K-Means, ML models for telecom churn
├── COM737-Sentiment-Analysis/
│   └── political_bias_detection.py # LSTM-based NLP classifier for political bias
├── COM740-Loan-Approval/
│   └── loan_approval.py           # ML pipeline for binary loan approval prediction
├── Dissertation-Eczema-Detection/
│   └── eczema_scratch_detection.py # Transformer models for IMU scratch detection
└── README.md
```

---

## Common Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
pip install tensorflow torch torchvision
pip install wordcloud nltk plotly
```

---

## Author

**Emmanuel Nonso Elege**
MSc Data Science · University of Ulster, Magee Campus
Registration No: B00994657
