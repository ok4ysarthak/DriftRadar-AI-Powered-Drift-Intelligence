# 🚨 DriftRadar: ML Data/Model Drift Monitor with Auto-Remediation Copilot  

![Docker](https://img.shields.io/badge/Docker-Desktop-blue?logo=docker)  
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)  
![React](https://img.shields.io/badge/React-Frontend-blue?logo=react)  
![Postgres](https://img.shields.io/badge/PostgreSQL-Database-316192?logo=postgresql)  
![LLM](https://img.shields.io/badge/Generative_AI-Llama_3-orange)  
![License](https://img.shields.io/badge/license-MIT-lightgrey)  

> **DriftRadar** is an **end-to-end MLOps platform** for monitoring and managing **data & model drift** in machine learning systems.  
> It integrates a **Generative AI Copilot** (powered by Llama 3) to **analyze drift incidents** and **suggest automated code-based fixes**.  

---

## 📌 Features  

- ✅ **Data & Model Drift Detection** – Statistical monitoring (PSI, KS-test, etc.)  
- ✅ **Interactive React Dashboard** – Visualize metrics, alerts, and performance  
- ✅ **Auto-Remediation Copilot** – LLM generates summaries + Python fixes  
- ✅ **Persistent Storage** – PostgreSQL for metrics & logs  
- ✅ **Full Containerization** – Deployable with Docker Compose  
- ✅ **End-to-End Pipeline** – Data ingestion → monitoring → remediation  

---

## 🏗️ System Architecture  

      ┌──────────────────┐
      │   React Frontend │
      │ (Dashboard + UI) │
      └─────────▲────────┘
                │
                ▼
      ┌──────────────────┐
      │   FastAPI API    │
      │ (Drift Detection │
      │  + LLM Orchestr.)│
      └─────────▲────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
    ┌────────────────┐ ┌──────────────────┐
    │ PostgreSQL DB │ │ LLM Service │
    │ (Metrics/Logs) │ │ (Ollama + Llama3)│
    └────────────────┘ └──────────────────┘



---

## ⚙️ Tech Stack  

- **Data Science**: Python, Pandas, NumPy, Scikit-learn, XGBoost, Evidently  
- **Backend**: FastAPI, Psycopg2  
- **Generative AI**: Ollama + Llama 3  
- **Frontend**: React, Tailwind CSS, Chart.js, Vite  
- **Database**: PostgreSQL  
- **DevOps**: Docker, Docker Compose  

---

## 🚀 Getting Started  

### 1️⃣ Prerequisites  
- [Docker Desktop](https://www.docker.com/products/docker-desktop)  
- [Node.js v18+](https://nodejs.org/en/download)  

---

### 2️⃣ Setup  

```bash
# Clone the repository
git clone https://github.com/your-username/DriftRadar.git
cd DriftRadar

📂 Dataset

Download the Telco Customer Churn dataset

Place it inside ./data/ as telco_churn.csv

📂 Environment Variables
Create .env file in root:

POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
POSTGRES_DB=driftradar

📂 Prepare Model & Data

python 01_train_baseline_model.py
python 02_inject_drift.py

3️⃣ Launch
docker-compose up --build

📊 Usage
✅ Ingest Data
Invoke-WebRequest -Method POST -Uri http://localhost:8000/ingest_batch/1
Invoke-WebRequest -Method POST -Uri http://localhost:8000/ingest_batch/2
# Repeat for batches 3–10

✅ View Dashboard

👉 http://localhost:3000

📈 Drift charts

⚠️ Alerts

🤖 "Generate Fix" button for LLM copilot

🤖 Auto-Remediation Example

Detected Drift:

Feature 'MonthlyCharges' PSI = 0.42  
Likely cause: pricing changes in recent data.


Generated Fix:

df["MonthlyCharges"] = df["MonthlyCharges"].clip(lower=20, upper=120)

🛠️ Development Notes

Backend Docs → http://localhost:8000/docs

Hot Reload → Enabled (uvicorn --reload)

DB Persistence → Docker volume driftradar_postgres_data

📌 Roadmap

 Multi-model drift monitoring

 Auto-remediation pipeline deployment

 Slack/Discord alerting

 Cloud LLM support (GPT-4, Claude, etc.)
