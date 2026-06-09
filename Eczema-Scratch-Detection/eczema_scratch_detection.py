# ============================================================
# Dissertation - Detecting Eczema Symptoms Using Classification Algorithms
# Focus: Scratch Detection Using Wearable Sensor Data & Deep Learning
# Author: Emmanuel Nonso Elege | University of Ulster, Magee Campus
# Dataset: COMBINED.csv (IMU sensor data: accelerometer, gyroscope, magnetometer)
# Models: Random Forest, SVM, Conformer Transformer, Standard Transformer
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# SECTION 1: LOAD DATASET
# ============================================================
df = pd.read_csv('COMBINED.csv')

feature_columns = [
    'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
    'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
    'totalAcceleration_x', 'totalAcceleration_y', 'totalAcceleration_z',
    'magnetometer_x', 'magnetometer_y', 'magnetometer_z'
]

X_raw = df[feature_columns].values
y = df['scratch_x'].values

# ============================================================
# SECTION 2: EDA
# ============================================================
plt.figure(figsize=(6, 4))
sns.countplot(x='scratch_x', data=df)
plt.title('Distribution of Scratching vs. Non-Scratching (Original)')
plt.xlabel('Scratching (1) vs. Non-Scratching (0)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Scratching', 'Scratching'])
plt.show()

# Correlation matrix
correlation_matrix = df[feature_columns + ['scratch_x']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature and Target Correlation Matrix')
plt.show()

# Train-test split for classical models
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# ============================================================
# SECTION 3: RANDOM FOREST MODEL
# ============================================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Scratching', 'Scratching'],
            yticklabels=['Non-Scratching', 'Scratching'])
plt.title('RF Confusion Matrix')
plt.show()

# ============================================================
# SECTION 4: SVM MODEL
# ============================================================
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Scratching', 'Scratching'],
            yticklabels=['Non-Scratching', 'Scratching'])
plt.title('SVM Confusion Matrix')
plt.show()

# ============================================================
# SECTION 5: FEATURE IMPORTANCE (Random Forest)
# ============================================================
importances = rf.feature_importances_
feature_importance = pd.Series(importances, index=feature_columns).sort_values(ascending=False)

fig, ax = plt.subplots()
feature_importance.plot.bar(ax=ax)
ax.set_title("Feature Importances (Random Forest)")
ax.set_ylabel("Mean decrease in impurity")
plt.show()

# ============================================================
# SECTION 6: SLIDING WINDOW PREPROCESSING
# ============================================================
window_size = 40
step_size = 20

windows = []
labels = []
for i in range(0, len(df) - window_size, step_size):
    windows.append(X_raw[i:i + window_size])
    labels.append(np.bincount(y[i:i + window_size].astype(int)).argmax())

X_windows = np.array(windows)
y_windows = np.array(labels)

def apply_windowing(X, y, window_size, step_size):
    """Apply sliding window with majority label voting."""
    wins, labs = [], []
    for i in range(0, len(X) - window_size, step_size):
        wins.append(X[i:i + window_size])
        window_labels = pd.Series(y[i:i + window_size]).fillna(-1).astype(int)
        valid = window_labels[window_labels != -1]
        labs.append(np.bincount(valid).argmax() if valid.size > 0 else 0)
    return np.array(wins), np.array(labs)

# ============================================================
# SECTION 7: TRANSFORMER MODEL DEFINITIONS
# ============================================================

class StandardTransformer(nn.Module):
    """Standard Transformer Encoder for scratch detection."""

    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2,
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)         # (window_size, batch, input_dim)
        x = self.transformer_encoder(x)
        x = x[-1]                       # Take the last time step
        x = self.dropout(x)
        return self.fc(x)


class Conformer(nn.Module):
    """Conformer: Conv + Transformer Encoder for local + global feature capture."""

    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2,
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)         # (batch, input_dim, window_size)
        x = self.conv(x)
        x = x.permute(2, 0, 1)         # (window_size, batch, input_dim)
        x = self.transformer_encoder(x)
        x = x[-1]
        x = self.dropout(x)
        return self.fc(x)


def train_transformer(model, X_train_t, y_train_t, X_test_t, y_test_t,
                      model_name, num_epochs=20):
    """Train and evaluate a Transformer model with early stopping."""
    train_dataset = TensorDataset(
        torch.tensor(X_train_t, dtype=torch.float32),
        torch.tensor(y_train_t, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_t, dtype=torch.float32),
        torch.tensor(y_test_t, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Weighted loss to handle class imbalance (scratching is rare)
    class_weights = torch.tensor([1.0, 10.0], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    model_path = f'best_{model_name.lower().replace(" ", "_")}_transformer.pt'

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                _, predicted = torch.max(model(inputs), 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(targets.cpu().numpy())

        val_f1 = f1_score(val_true, val_preds, average='weighted')
        print(f"Epoch {epoch + 1}/{num_epochs}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(model_path))
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            _, predicted = torch.max(model(inputs), 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(targets.cpu().numpy())

    print(f"\n{model_name} Final Classification Report:")
    print(classification_report(val_true, val_preds))

    cm = confusion_matrix(val_true, val_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Scratching', 'Scratching'],
                yticklabels=['Non-Scratching', 'Scratching'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    return val_f1


# ============================================================
# SECTION 8: TRAIN TRANSFORMER MODELS
# ============================================================
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_windows, y_windows, test_size=0.2, random_state=42
)

input_dim = X_windows.shape[2]   # 12 sensor channels
num_classes = 2

conformer_model = Conformer(input_dim, num_classes)
print("Training Conformer:")
conformer_f1 = train_transformer(
    conformer_model, X_train_t, y_train_t, X_test_t, y_test_t,
    model_name="Conformer", num_epochs=20
)

standard_transformer_model = StandardTransformer(input_dim, num_classes)
print("\nTraining Standard Transformer:")
standard_transformer_f1 = train_transformer(
    standard_transformer_model, X_train_t, y_train_t, X_test_t, y_test_t,
    model_name="Standard Transformer", num_epochs=20
)

# ============================================================
# SECTION 9: INFERENCE ON MULTIPLE VALIDATION DATASETS
# ============================================================
# Load validation datasets
df_val   = pd.read_csv('VALID.csv')
df_flat  = pd.read_csv('FLAT.csv')
df_none  = pd.read_csv('NONE.csv')
df_touch = pd.read_csv('TOUCH.csv')
df_slow  = pd.read_csv('SLOW.csv')

def predict_on_dataset(model, X_windows):
    """Run inference on windowed sensor data."""
    dataset = TensorDataset(torch.tensor(X_windows, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    preds = []
    with torch.no_grad():
        for (inputs,) in loader:
            _, predicted = torch.max(model(inputs), 1)
            preds.extend(predicted.cpu().numpy())
    return preds

def count_scratching_actions(predictions):
    """Count distinct scratch events (consecutive 1-blocks)."""
    scratch_actions = 0
    in_scratch = False
    for pred in predictions:
        if pred == 1 and not in_scratch:
            scratch_actions += 1
            in_scratch = True
        elif pred == 0:
            in_scratch = False
    return scratch_actions

# Apply windowing and predict for each test scenario
datasets_raw = {
    "VALID": (df_val, 'VALID.csv'),
    "FLAT":  (df_flat, 'FLAT.csv'),
    "NONE":  (df_none, 'NONE.csv'),
    "TOUCH": (df_touch, 'TOUCH.csv'),
    "SLOW":  (df_slow, 'SLOW.csv'),
}

datasets_windowed = {}
for name, (df_d, _) in datasets_raw.items():
    X_w, y_w = apply_windowing(df_d[feature_columns].values,
                                df_d['scratch_x'].values, window_size, step_size)
    datasets_windowed[name] = (X_w, y_w)

# Load best saved models
conformer_model.load_state_dict(torch.load('best_conformer_transformer.pt'))
conformer_model.eval()
standard_transformer_model.load_state_dict(
    torch.load('best_standard_transformer_transformer.pt'))
standard_transformer_model.eval()

# Predict and report
conformer_scratch_counts = {}
standard_transformer_scratch_counts = {}

for name, (X_w, y_w) in datasets_windowed.items():
    conf_preds = predict_on_dataset(conformer_model, X_w)
    std_preds  = predict_on_dataset(standard_transformer_model, X_w)

    conf_actions = count_scratching_actions(conf_preds)
    std_actions  = count_scratching_actions(std_preds)

    conformer_scratch_counts[name] = conf_actions
    standard_transformer_scratch_counts[name] = std_actions

    print(f"\n{name} - Conformer scratch actions: {conf_actions}")
    print(f"{name} - Standard Transformer scratch actions: {std_actions}")

# ============================================================
# SECTION 10: VISUALISE SCRATCH COUNTS
# ============================================================
scratch_counts_df = pd.DataFrame({
    'Conformer': conformer_scratch_counts,
    'Standard Transformer': standard_transformer_scratch_counts,
})

ax = scratch_counts_df.plot(kind='bar', figsize=(12, 6))
ax.set_xlabel('Dataset', fontsize=20)
ax.set_ylabel('Number of Scratching Actions', fontsize=20)
ax.set_title('Predicted Scratching Actions by Dataset and Model', fontsize=25)
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.show()

# ============================================================
# SECTION 11: WINDOW SIZE / STEP SIZE SENSITIVITY ANALYSIS
# ============================================================
window_sizes = [20, 40, 60, 80]
step_sizes   = [10, 20, 30, 40]
results = []

for ws in window_sizes:
    for ss in step_sizes:
        print(f"\nWindow: {ws}, Step: {ss}")
        X_w_train, y_w_train = apply_windowing(X_raw, y, ws, ss)
        X_w_val, y_w_val = apply_windowing(
            df_val[feature_columns].values, df_val['scratch_x'].values, ws, ss
        )

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_w_train, y_w_train, test_size=0.2, random_state=42
        )
        input_dim_ws = X_w_train.shape[2]

        conf_m = Conformer(input_dim_ws, num_classes)
        conf_f1 = train_transformer(conf_m, X_tr, y_tr, X_te, y_te,
                                    model_name=f"Conformer_ws{ws}_ss{ss}",
                                    num_epochs=5)

        std_m = StandardTransformer(input_dim_ws, num_classes)
        std_f1 = train_transformer(std_m, X_tr, y_tr, X_te, y_te,
                                   model_name=f"StdTransformer_ws{ws}_ss{ss}",
                                   num_epochs=5)

        results.append({
            'window_size': ws, 'step_size': ss,
            'conformer_f1': conf_f1, 'standard_transformer_f1': std_f1
        })

results_df = pd.DataFrame(results)

# Heatmaps
for model_col, title in [('conformer_f1', 'Conformer'),
                           ('standard_transformer_f1', 'Standard Transformer')]:
    pivot = results_df.pivot(index='window_size', columns='step_size', values=model_col)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt=".3f")
    plt.title(f'{title} F1-scores by Window & Step Size')
    plt.xlabel('Step Size')
    plt.ylabel('Window Size')
    plt.show()
