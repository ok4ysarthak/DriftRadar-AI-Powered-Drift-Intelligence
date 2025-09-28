import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit

def data_clean_and_preprocess(radar: pd.DataFrame):
    if 'Unnamed: 0' in radar.columns:
        radar = radar.drop('Unnamed: 0', axis=1)
    if 'customerID' in radar.columns:
        radar = radar.drop('customerID', axis=1)
    radar['TotalCharges'] = pd.to_numeric(radar['TotalCharges'], errors='coerce')
    radar.dropna(subset=['TotalCharges'], inplace=True)
    if 'Churn' in radar.columns:
        radar['Churn'] = radar['Churn'].map({'Yes': 1, 'No': 0})
    binary_map = {'yes': 1, 'no': 0, 'no internet service': 0, 'no phone service': 0,
                  'true': 1, 'false': 0, '1': 1, '0': 0}
    for col in ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling']:
        if col in radar.columns:
            radar[col] = radar[col].astype(str).str.lower().map(binary_map)
    return radar

def inject_drift(input_path, output_dir, num_batches=10, batch_size=700):
    print("Starting drift injection...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    df = pd.read_csv(input_path)
    df = data_clean_and_preprocess(df)
    df.dropna(subset=['Churn'], inplace=True)

    # Use stratified sampling to ensure both Churn and No Churn classes are in every batch
    splitter = StratifiedShuffleSplit(n_splits=num_batches, test_size=batch_size, random_state=42)
    
    for i, (_, batch_indices) in enumerate(splitter.split(df, df['Churn'])):
        batch = df.iloc[batch_indices].copy()

        # Inject numerical drift: increase MonthlyCharges over time
        if i > 5:
            batch['MonthlyCharges'] *= (1 + (i - 5) * 0.1)
            print(f"Batch {i+1}: Injected numerical drift")

        # Inject categorical drift: change contract distribution
        if i > 7 and 'Contract' in batch.columns:
            idx = batch[batch['Contract'] == 'One year'].sample(frac=0.2, replace=False).index
            batch.loc[idx, 'Contract'] = 'Month-to-month'
            print(f"Batch {i+1}: Injected categorical drift")

        batch.to_csv(os.path.join(output_dir, f"batch_{i+1}.csv"), index=False)
        print(f"Saved batch {i+1} to {os.path.join(output_dir, f'batch_{i+1}.csv')} (Churn distribution: {batch['Churn'].value_counts().to_dict()})")
        
    print("✅ Drift batches ready.")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")
    OUTPUT_BATCHES_DIR = os.path.join(PROJECT_ROOT, "scripts", "data_streams")
    inject_drift(ORIGINAL_DATA_PATH, OUTPUT_BATCHES_DIR)
