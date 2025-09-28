import pandas as pd
import numpy as np
import joblib
import os
import psycopg2
from sklearn.metrics import f1_score, roc_auc_score
from dotenv import load_dotenv
import warnings, sys

warnings.filterwarnings('ignore')
load_dotenv()

# --- Project Root & Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_artifacts", "churn_model.pkl")
DATA_STREAMS_DIR = os.path.join(PROJECT_ROOT, "scripts", "data_streams")
REFERENCE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")

# --- DB Config ---
DB_HOST, DB_NAME, DB_USER, DB_PASS = (
    os.getenv("DB_HOST"), os.getenv("DB_NAME"),
    os.getenv("DB_USER"), os.getenv("DB_PASS")
)

def connect_db():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)

# (rest of your drift + monitoring code stays same, just using new paths)


def log_metric_to_db(conn, batch_id, feature_name, drift_score, is_drifted, alert_message, metric_type):
    """Logs a single metric to the PostgreSQL database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO drift_metrics (batch_id, feature_name, drift_score, is_drifted, alert_message, metric_type)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (batch_id, feature_name, drift_score, is_drifted, alert_message, metric_type))
    conn.commit()
    cursor.close()

def calculate_psi_numerical(expected_dist, actual_dist, n_bins=10):
    """Calculates PSI for numerical data by binning."""
    expected_dist = pd.to_numeric(expected_dist, errors="coerce").fillna(0)
    actual_dist = pd.to_numeric(actual_dist, errors="coerce").fillna(0)

    bins = np.linspace(min(expected_dist.min(), actual_dist.min()), 
                       max(expected_dist.max(), actual_dist.max()), n_bins)
    
    expected_counts, _ = np.histogram(expected_dist, bins=bins)
    actual_counts, _ = np.histogram(actual_dist, bins=bins)
    
    expected_prop = expected_counts / len(expected_dist)
    actual_prop = actual_counts / len(actual_dist)
    
    expected_prop = np.where(expected_prop == 0, 1e-6, expected_prop)
    actual_prop = np.where(actual_prop == 0, 1e-6, actual_prop)
    
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    return psi

def calculate_psi_categorical(expected_dist, actual_dist):
    """Calculates PSI for categorical data."""
    expected_dist = expected_dist.astype(str).fillna("MISSING")
    actual_dist = actual_dist.astype(str).fillna("MISSING")

    all_categories = pd.concat([expected_dist, actual_dist]).unique()
    expected_counts = expected_dist.value_counts().reindex(all_categories, fill_value=0)
    actual_counts = actual_dist.value_counts().reindex(all_categories, fill_value=0)
    
    expected_prop = expected_counts / len(expected_dist)
    actual_prop = actual_counts / len(actual_dist)
    
    expected_prop = np.where(expected_prop == 0, 1e-6, expected_prop)
    actual_prop = np.where(actual_prop == 0, 1e-6, actual_prop)
    
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    return psi

def data_clean_and_preprocess(radar: pd.DataFrame):
    if 'Unnamed: 0' in radar.columns:
        radar = radar.drop('Unnamed: 0', axis=1)
    if 'customerID' in radar.columns:
        radar = radar.drop('customerID', axis=1)

    # Convert 'TotalCharges' to numeric, fill missing instead of dropping rows
    if 'TotalCharges' in radar.columns:
        radar['TotalCharges'] = pd.to_numeric(radar['TotalCharges'], errors='coerce').fillna(0)

    # Handle 'Churn' robustly
    if 'Churn' in radar.columns:
        if pd.api.types.is_numeric_dtype(radar['Churn']):
            radar['Churn'] = radar['Churn'].fillna(0).astype(int)
        else:
            radar['Churn'] = (
                radar['Churn'].astype(str).str.strip().str.lower().map({
                    'yes': 1, 'no': 0,
                    '1': 1, '0': 0,
                    'true': 1, 'false': 0
                })
            )
            radar['Churn'] = radar['Churn'].fillna(0).astype(int)

    # Binary columns mapping
    binary_map = {
        'yes': 1, 'no': 0,
        'no internet service': 0, 'no phone service': 0,
        'true': 1, 'false': 0, '1': 1, '0': 0
    }
    
    binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 
                   'StreamingMovies', 'PaperlessBilling']
    
    for col in binary_cols:
        if col in radar.columns:
            radar[col] = radar[col].astype(str).str.strip().str.lower().map(binary_map).fillna(0).astype(int)
            
    return radar

def monitor_batch(batch_id: int, model_pipeline, reference_df: pd.DataFrame, current_df: pd.DataFrame, conn):
    print(f"\n--- Monitoring Batch {batch_id} ---")
    
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
    binary_features = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'PaperlessBilling']
    
    # --- 1. Data Drift Detection ---
    print("  Detecting data drift...")
    for feature in numerical_features:
        if feature in current_df.columns and feature in reference_df.columns:
            psi_score = calculate_psi_numerical(reference_df[feature], current_df[feature])
            is_drifted = bool(psi_score > 0.25)
            alert_message = f"Numerical PSI: {psi_score:.4f}"
            if is_drifted:
                print(f"  🚨 ALERT: Significant drift in '{feature}'. {alert_message}")
            log_metric_to_db(conn, batch_id, feature, float(psi_score), is_drifted, alert_message, 'data_drift_psi_num')

    for feature in categorical_features:
        if feature in current_df.columns and feature in reference_df.columns:
            psi_score = calculate_psi_categorical(reference_df[feature], current_df[feature])
            is_drifted = bool(psi_score > 0.25)
            alert_message = f"Categorical PSI: {psi_score:.4f}"
            if is_drifted:
                print(f"  🚨 ALERT: Significant drift in '{feature}'. {alert_message}")
            log_metric_to_db(conn, batch_id, feature, float(psi_score), is_drifted, alert_message, 'data_drift_psi_cat')
    
    for feature in binary_features:
        if feature in current_df.columns and feature in reference_df.columns:
            psi_score = calculate_psi_numerical(reference_df[feature], current_df[feature])
            is_drifted = bool(psi_score > 0.25)
            alert_message = f"Binary PSI: {psi_score:.4f}"
            if is_drifted:
                print(f"  🚨 ALERT: Significant drift in '{feature}'. {alert_message}")
            log_metric_to_db(conn, batch_id, feature, float(psi_score), is_drifted, alert_message, 'data_drift_psi_bin')

    # --- 2. Model Performance Monitoring ---
    print("  Monitoring model performance...")
    if 'Churn' not in current_df.columns:
        print(f"  ⚠️ Batch {batch_id} has no Churn column. Skipping model performance.")
        return
    
    X_current = current_df.drop('Churn', axis=1, errors='ignore')
    y_true = current_df['Churn']
    
    try:
        y_pred = model_pipeline.predict(X_current)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"  ⚠️ Model prediction failed: {e}")
        return
    
    log_metric_to_db(conn, batch_id, 'model', float(auc), False, f"Model AUC: {auc:.4f}", 'model_perf_auc')
    log_metric_to_db(conn, batch_id, 'model', float(f1), False, f"Model F1: {f1:.4f}", 'model_perf_f1')
    print(f"  ✅ Model Performance: F1={f1:.4f}, AUC={auc:.4f}")

if __name__ == "__main__":
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print("Model pipeline loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model pipeline not found at '{MODEL_PATH}'. Please run '01_train_baseline_model.py' first.")
        sys.exit(1)
        
    try:
        reference_data = pd.read_csv(REFERENCE_DATA_PATH)
        reference_data = data_clean_and_preprocess(reference_data)
    except FileNotFoundError:
        print(f"Error: Reference dataset not found at '{REFERENCE_DATA_PATH}'. Please ensure it is in the project directory.")
        sys.exit(1)

    conn = None
    try:
        conn = connect_db()
        print("Database connection successful.")
        
        batch_files = sorted([f for f in os.listdir(DATA_STREAMS_DIR) if f.startswith('batch_') and f.endswith('.csv')])
        
        if not batch_files:
            print("No data batches found. Please run '02_inject_drift.py' first.")
            sys.exit(1)
            
        for batch_file in batch_files:
            batch_id = int(batch_file.split('_')[1].split('.')[0])
            batch_df = pd.read_csv(os.path.join(DATA_STREAMS_DIR, batch_file))
            
            batch_df = data_clean_and_preprocess(batch_df)
            
            if batch_df.empty:
                print(f"--- Warning: Batch {batch_id} is empty after preprocessing. Skipping monitoring. ---")
                log_metric_to_db(conn, batch_id, 'data_quality', 0, True, 
                                 'Batch is empty after preprocessing.', 'data_quality_issue')
                continue

            monitor_batch(batch_id, model_pipeline, reference_data, batch_df, conn)
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
