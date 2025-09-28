import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)

warnings.filterwarnings('ignore')

# --- Project Root & Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "model_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "churn_model.pkl")

# --- Preprocessing ---
def data_clean_and_preprocess(radar: pd.DataFrame):
    if 'Unnamed: 0' in radar.columns:
        radar = radar.drop('Unnamed: 0', axis=1)
    if 'customerID' in radar.columns:
        radar = radar.drop('customerID', axis=1)

    radar['TotalCharges'] = pd.to_numeric(radar['TotalCharges'], errors='coerce')
    radar.dropna(subset=['TotalCharges'], inplace=True)

    if 'Churn' in radar.columns:
        radar['Churn'] = radar['Churn'].map({'Yes': 1, 'No': 0})

    binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'PaperlessBilling']
    for col in binary_cols:
        if col in radar.columns:
            radar[col] = radar[col].astype(str).str.lower().map({
                'yes': 1, 'no': 0, 'true': 1, 'false': 0,
                '1': 1, '0': 0, 'no internet service': 0, 'no phone service': 0
            })
    return radar

# --- Load Dataset ---
try:
    radar = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at '{DATA_PATH}'")
    exit()

radar = data_clean_and_preprocess(radar)
radar.dropna(subset=['Churn'], inplace=True)

# --- Features ---
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
binary_features = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'PaperlessBilling']

X = radar.drop('Churn', axis=1)
y = radar['Churn']

# --- Handle Imbalance ---
neg, pos = np.bincount(y.astype(int))
scale_pos_weight = neg / pos
print(f"Class imbalance: {neg} neg, {pos} pos → scale_pos_weight={scale_pos_weight:.2f}")

# --- Pipeline ---
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('bin', OneHotEncoder(handle_unknown='ignore'), binary_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        use_label_encoder=False, scale_pos_weight=scale_pos_weight, random_state=42
    ))
])

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model_pipeline.fit(X_train, y_train)

# --- Save Model ---
joblib.dump(model_pipeline, ARTIFACT_PATH)
print(f"✅ Model saved at {ARTIFACT_PATH}")

# --- Evaluation ---
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
