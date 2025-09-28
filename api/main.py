# api/main.py

import sys
import os

# Add the project root to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import pandas as pd
import warnings

from src.drift_monitor import monitor_batch, connect_db
from src.models import DriftMetric

warnings.filterwarnings('ignore')

app = FastAPI(
    title="DriftRadar API",
    description="API for ML Data/Model Drift Monitor + Auto-Remediation Copilot",
    version="0.1.0",
)

origins = ["http://localhost:3000", "http://localhost"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Paths ---
CONTAINER_ROOT = "/app"
MODEL_PATH = os.path.join(CONTAINER_ROOT, "model_artifacts", "churn_model.pkl") # <-- CORRECTED
REFERENCE_DATA_PATH = os.path.join(CONTAINER_ROOT, "data", "telco_churn.csv")
DATA_STREAMS_DIR = os.path.join(CONTAINER_ROOT, "scripts", "data_streams") # <-- CORRECTED

# --- Load artifacts once on startup ---
try:
    model_pipeline = joblib.load(MODEL_PATH)
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    reference_data['TotalCharges'] = pd.to_numeric(reference_data['TotalCharges'], errors='coerce')
    reference_data.dropna(inplace=True)
    print("Model and reference data loaded successfully on API startup.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find required files. Please ensure they exist: {e}")
    sys.exit(1)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the DriftRadar API!"}

@app.get("/metrics", response_model=List[DriftMetric])
def get_metrics():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM drift_metrics ORDER BY timestamp DESC")
        metrics = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        result = [dict(zip(column_names, row)) for row in metrics]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

@app.post("/ingest_batch/{batch_id}")
def ingest_new_batch(batch_id: int):
    try:
        batch_path = os.path.join(DATA_STREAMS_DIR, f'batch_{batch_id}.csv')
        if not os.path.exists(batch_path):
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found at {batch_path}")
        
        current_data = pd.read_csv(batch_path)
        current_data['TotalCharges'] = pd.to_numeric(current_data['TotalCharges'], errors='coerce')
        current_data.dropna(inplace=True)
        current_data['Churn'] = 0
        
        conn = connect_db()
        monitor_batch(batch_id, model_pipeline, reference_data, current_data, conn)
        conn.close()
        
        return {"status": "success", "message": f"Batch {batch_id} monitored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/generate_fix/{batch_id}")
async def generate_remediation_fix(batch_id: int):
    """
    Triggers the LLM to generate a root cause analysis and a remediation fix.
    """
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT feature_name, drift_score, alert_message FROM drift_metrics WHERE batch_id = %s AND is_drifted = TRUE", (batch_id,))
        drifting_features = cursor.fetchall()
        
        if not drifting_features:
            return {"message": f"No significant drift detected for batch {batch_id}. No fix required."}

        prompt_components = []
        for feature, score, message in drifting_features:
            prompt_components.append(f"Feature '{feature}' has drifted. Drift score: {score:.4f}. Alert: {message}")
        
        incident_summary = "\n".join(prompt_components)
        
        system_prompt = (
            "You are a Senior MLOps Engineer. Your task is to analyze a data drift incident. "
            "Provide a clear, concise root cause analysis (less than 100 words) and a Python code snippet to fix the issue. "
            "Assume the data is in a pandas DataFrame called 'df'. "
            "Provide the Python code in a code block. Do not provide extra words in the code block. "
            "Your output should be a JSON object with two keys: 'root_cause' (string) and 'remediation_code' (string)."
        )
        
        user_prompt = f"Analyze the following data drift incident and propose a fix:\n\n{incident_summary}"
        
        try:
            response = ollama_client.chat(model='llama3', messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])
            llm_output = response['message']['content']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM API call failed: {str(e)}. Is Ollama running?")

        return {"llm_response": llm_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()