# ============================================================
# COM735 - Churn Analysis: Optimizing Customer Retention Strategies in Telecom
# Author: Emmanuel Nonso Elege | University of Ulster, Magee Campus
# Dataset: customer_churn_telecom_services.csv (~7,043 records)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# ---- Load and Preprocess Dataset ----
df = pd.read_csv("customer_churn_telecom_services.csv")

# Handle missing values
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert target variable to numeric
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ---- Exploratory Data Analysis (EDA) ----
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Churn'], palette='coolwarm')
plt.title("Churn Distribution")
plt.show()

# Distribution of numerical features
df.hist(figsize=(12, 10), bins=20, color='red', edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()

# Define visualizations
visualizations = [
    ("tenure", "Churn", "Churn Rate by Customer Tenure"),
    ("MonthlyCharges", "Churn", "Churn Rate by Monthly Charges"),
    ("TotalCharges", "Churn", "Churn Rate by Total Charges"),
    ("Contract", "Churn", "Churn Rate by Contract Type"),
    ("PaymentMethod", "Churn", "Churn Rate by Payment Method"),
    ("InternetService", "Churn", "Churn by Internet Service Type"),
    ("TechSupport", "Churn", "Churn by Tech Support"),
    ("OnlineSecurity", "Churn", "Churn by Online Security"),
    ("OnlineBackup", "Churn", "Churn by Online Backup"),
    ("DeviceProtection", "Churn", "Churn by Device Protection"),
    ("StreamingTV", "Churn", "Churn by Streaming TV"),
    ("StreamingMovies", "Churn", "Churn by Streaming Movies"),
    ("PaperlessBilling", "Churn", "Churn by Paperless Billing"),
    ("MultipleLines", "Churn", "Churn by Multiple Lines"),
    ("gender", "Churn", "Churn by Gender"),
    ("SeniorCitizen", "Churn", "Churn by Senior Citizen"),
    ("Partner", "Churn", "Churn by Partner"),
    ("Dependents", "Churn", "Churn by Dependents"),
]

for feature, target, title in visualizations:
    plt.figure(figsize=(8, 5))
    if df[feature].dtype == 'O' or df[feature].dtype.name == 'category':
        ax = plt.gca()
        sns.barplot(x=df[feature], y=df[target], palette="coolwarm", ax=ax)
        plt.xticks(rotation=45)
        ax.set(yticklabels=[])
    else:
        sns.kdeplot(df[df[target] == 1][feature], label="Churn", fill=True)
        sns.kdeplot(df[df[target] == 0][feature], label="No Churn", fill=True)
        plt.yticks([])
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel("Churn Rate" if df[feature].dtype.name == 'category' else "Density")
    plt.legend()
    plt.show()

# ---- Feature Engineering ----
# Encode categorical variables
df_encoded_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in df_encoded_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Remove highly correlated features (threshold = 0.80)
correlation_threshold = 0.80
corr_matrix = df.corr()
high_corr_pairs = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            high_corr_pairs.add((corr_matrix.columns[i], corr_matrix.columns[j]))

columns_to_drop = {col2 for col1, col2 in high_corr_pairs}
df_reduced = df.drop(columns=columns_to_drop)
print(f"Dropped: {columns_to_drop}")
print(f"Remaining: {list(df_reduced.columns)}")

# ---- Model Training ----
X = df.drop(columns=['Churn'])
y = df['Churn']

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

sns.countplot(x=y_resampled, palette='coolwarm')
plt.title("Churn Distribution After SMOTE")
plt.show()

# Train and evaluate models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ---- ROC Curve ----
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ---- Feature Importance (Random Forest) ----
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
feature_importances = pd.Series(
    rf_model.feature_importances_, index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10], palette="coolwarm")
plt.title("Top 10 Feature Importances for Churn Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# ---- Customer Segmentation (K-Means Clustering) ----
segmentation_features = ['MonthlyCharges', 'tenure', 'TotalCharges']
X_segmentation = df[segmentation_features]

kmeans = KMeans(n_clusters=3, random_state=42)
df['CustomerSegment'] = kmeans.fit_predict(X_segmentation)

segment_labels = {
    0: "Mid-Spending, Medium-Term Customers",
    1: "High-Spending, Long-Term Customers",
    2: "Low-Spending, Short-Term Customers",
}
df['CustomerSegmentLabel'] = df['CustomerSegment'].map(segment_labels)

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='MonthlyCharges', y='tenure',
    hue='CustomerSegmentLabel', data=df, palette='viridis'
)
plt.title('Customer Segmentation')
plt.xlabel('Monthly Charges')
plt.ylabel('Tenure (Months)')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='CustomerSegmentLabel', y='TotalCharges', data=df, palette='Set3')
plt.title('Total Charges by Customer Segment')
plt.xticks(rotation=45, ha='right')
plt.show()

# Pie chart
plt.figure(figsize=(6, 6))
segment_counts = df['CustomerSegmentLabel'].value_counts()
plt.pie(
    segment_counts, labels=segment_counts.index,
    autopct='%1.1f%%', startangle=90,
    colors=['lightblue', 'lightgreen', 'lightcoral']
)
plt.title('Customer Segment Distribution')
plt.show()
