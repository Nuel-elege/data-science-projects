# ============================================================
# COM740 - Loan Approval Prediction
# Author: Emmanuel Nonso Elege | University of Ulster, Magee Campus
# Dataset: loan_data.csv (~45,000 records)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE

# ---- Load Dataset ----
df = pd.read_csv("loan_data.csv")
print(df.info())

# Identify column types
categorical_cols = [var for var in df.columns if df[var].dtypes == 'object']
numerical_cols = [var for var in df.columns if df[var].dtypes != 'object']
print(f'Categorical columns: {categorical_cols}')
print(f'Numerical columns: {numerical_cols}')

# ---- Univariate Analysis ----
def univariate_analysis_boxplot(data, column, title):
    plt.figure(figsize=(10, 2))
    sns.boxplot(x=data[column], color='coral')
    plt.title(f'{title} Boxplot')
    plt.tight_layout()
    plt.show()
    print(f'Summary Stats for {title}:\n', data[column].describe())

for column in numerical_cols:
    univariate_analysis_boxplot(df, column, column.replace('_', ' '))

def univariate_analysis_hist(data, columns):
    plt.figure(figsize=(10, 14))
    for i, column in enumerate(columns, 1):
        plt.subplot(5, 2, i)
        sns.histplot(data[column], kde=True, bins=30, color='coral')
        plt.title(f'{column.replace("_", " ")} Distribution with KDE')
        plt.xlabel(column.replace('_', ' '))
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

univariate_analysis_hist(df, numerical_cols)

# ---- Bivariate Analysis ----
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Loan Status by Categorical Features", fontsize=18)

titles = ["Gender", "Education Level", "Home Ownership", "Loan Intent", "Previous Loan Defaults"]

for i, (feature, title) in enumerate(zip(categorical_cols, titles)):
    row, col = divmod(i, 3)
    sns.countplot(data=df, x=feature, hue='loan_status', ax=axes[row, col], palette='muted')
    axes[row, col].set_title(f"Loan Status by {title}")
    axes[row, col].set_xlabel(title)
    axes[row, col].set_ylabel("Count")
    axes[row, col].legend(title='Loan Status', labels=['0 = Rejected', '1 = Approved'])
    if feature == 'loan_intent':
        axes[row, col].tick_params(axis='x', rotation=45)

if len(categorical_cols) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Density plots for numerical features
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('Numerical Features vs Loan Status (Density Plots)', fontsize=16)
axes = axes.flatten()

plot_index = 0
for col in numerical_cols:
    if col == 'loan_status':
        continue
    sns.kdeplot(data=df, x=col, hue='loan_status', ax=axes[plot_index],
                fill=True, common_norm=False, palette='muted')
    axes[plot_index].set_title(f'{col} vs Loan Status')
    axes[plot_index].set_xlabel(col)
    axes[plot_index].set_ylabel('Density')
    plot_index += 1

for i in range(plot_index, len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Target label proportions
colors = ['steelblue', 'sandybrown']
label_prop = df['loan_status'].value_counts()
plt.pie(label_prop.values, labels=['Rejected (0)', 'Approved (1)'],
        autopct='%.2f', colors=colors)
plt.title('Target Label Proportions')
plt.show()

print("Missing values:\n", df.isnull().sum())

# ---- Feature Engineering ----
# Handle outliers
median_age = df['person_age'].median()
df['person_age'] = df['person_age'].apply(lambda x: median_age if x > 100 else x)

median_exp = df['person_emp_exp'].median()
df['person_emp_exp'] = df['person_emp_exp'].apply(lambda x: median_exp if x > 80 else x)

# Binary encoding
df['person_gender'] = df['person_gender'].map({'female': 0, 'male': 1})
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

# Ordinal encoding for education
education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
df['person_education'] = df['person_education'].map(education_order)

# One-hot encoding for home ownership and loan intent
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

# Log transformation for skewed features
skewed_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                  'cb_person_cred_hist_length', 'credit_score', 'loan_percent_income']

fig, axes = plt.subplots(len(skewed_columns), 2, figsize=(12, len(skewed_columns) * 4))
fig.suptitle("Boxplots Before and After Log Transformation", fontsize=16, y=1.02)

for i, col in enumerate(skewed_columns):
    axes[i, 0].boxplot(df[col], vert=False, patch_artist=True,
                       boxprops=dict(facecolor='skyblue'))
    axes[i, 0].set_title(f"{col} - Before")

    df[col] = np.log1p(df[col])

    axes[i, 1].boxplot(df[col], vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen'))
    axes[i, 1].set_title(f"{col} - After")

plt.tight_layout()
plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Remove highly correlated features
correlation_threshold = 0.80
high_corr_pairs = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            high_corr_pairs.add((corr_matrix.columns[i], corr_matrix.columns[j]))

columns_to_drop = {col2 for col1, col2 in high_corr_pairs}
df_reduced = df.drop(columns=columns_to_drop)
print(f"Dropped: {columns_to_drop}")
print(f"Remaining: {list(df_reduced.columns)}")

# Correlation with target
target_corr = corr_matrix[['loan_status']].sort_values(by='loan_status', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=target_corr['loan_status'], y=target_corr.index,
            palette='coolwarm', hue=target_corr.index, dodge=False, legend=False)
plt.title('Correlation with Loan Status', fontsize=16)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# Select features with high correlation to target
threshold = 0.1
high_corr_features = corr_matrix.index[abs(corr_matrix["loan_status"]) > threshold].tolist()
high_corr_features.remove("loan_status")
print("Selected features:", high_corr_features)

X = df[high_corr_features]
Y = df["loan_status"]

# ---- Model Training ----
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale features
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply SMOTE for class imbalance
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Class distribution before SMOTE:", y_train.value_counts())
print("Class distribution after SMOTE:", y_train_smote.value_counts())

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_val, y_val)
    accuracy = accuracy_score(y_val, y_val_pred)

    results.append({'Model': name, 'Train Score': train_score,
                    'Test Score': test_score, 'Accuracy Score': accuracy})

    print(f"Classification Report for {name}:\n")
    print(classification_report(y_val, y_val_pred))

    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print("\n" + "="*60 + "\n")

results_df = pd.DataFrame(results)
print("Model Performance Table:")
print(results_df)

# ---- Cross-Validation ----
cv_results = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
    cv_results[model_name] = cv_scores
    print(f"{model_name}:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f}")
    print(f"  Std Dev: {np.std(cv_scores):.4f}\n")

plt.figure(figsize=(10, 6))
for model_name, cv_scores in cv_results.items():
    plt.plot(range(1, 11), cv_scores, marker='o', linestyle='--', label=model_name)
plt.title('10-Fold Cross-Validation Scores')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- ROC Curves ----
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.title("ROC Curve Comparison", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Best Model & Feature Importance ----
best_model_row = results_df.loc[results_df['Accuracy Score'].idxmax()]
best_model_name = best_model_row['Model']
best_model_accuracy = best_model_row['Accuracy Score']
print(f"\nBest Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")

best_model = models[best_model_name]
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns, 'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df,
                palette='viridis', hue=feature_importance_df.index)
    plt.title(f'Feature Importances - {best_model_name}', fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
