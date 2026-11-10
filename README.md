# 🎓 Bikash Kumar Singh - Nexvia Technologies AI/ML Capstone Projects

**Intern ID:** NEX-INT-2026-0134  
**Batch:** October 2026  
**Specialization:** AI/Machine Learning

---

## 📁 Project Structure

This repository contains both capstone projects completed during the 30-day Nexvia Technologies AI/ML internship:

### 🔹 AIML-01: Real-Time Financial Fraud Detection Platform

End-to-end MLOps pipeline with Kafka streaming, XGBoost + GraphSAGE ensemble, Feast Feature Store, Kubernetes deployment, Evidently AI drift monitoring, and Streamlit human-in-the-loop dashboard.

**Cloud Deployments (Live):**
- 🔗 **Live API:** https://aiml-01-fraud-platform.onrender.com/docs
-  **Live Dashboard:** https://aiml-01-fraud-platform.streamlit.app

### 🔹 AIML-02: AI-Assisted Medical Imaging and Diagnosis System

Deep learning platform for X-ray/CT/MRI scan analysis with CNN models, Grad-CAM explainability heatmaps, and doctor review workflow.

**Cloud Deployments (Live):**
- 🔗 **Live API:** https://aiml-02-medical-imaging-assist.onrender.com/docs
- 🔗 **Live Demo:** https://aiml-02-medical-imaging-assist.onrender.com

---

## 📊 30-Day Curriculum Completion

✅ **Module 1:** Python, NumPy, Pandas, Data Cleaning & EDA  
✅ **Module 2:** Classical ML (Regression, Classification, Ensembles)  
✅ **Module 3:** Advanced ML & Deep Learning (XGBoost, PyTorch, GNN)  
✅ **Module 4:** MLOps (FastAPI, Git, Monitoring, Drift Detection)  
✅ **Module 5:** Capstone Projects (This Repository)

---

## 🚀 Quick Start


```bash
### AIML-01: Real-Time Financial Fraud Detection Platform
# 1. Python 3.11 venv (Recommended for dependency compatibility)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies (FastAPI, XGBoost, Kafka, Evidently, Streamlit, etc.)
pip install -r requirements.txt

# 3. Ensure the trained model artifact is present (already in repo)
# Note: For the drift monitor to run locally, ensure `train_transaction.csv` 
# is in the root directory (excluded from Git due to size).

# 4. Start the real-time infrastructure (Zookeeper + Kafka + FastAPI)
# Note: If port 8000 is occupied by a local Kubernetes deployment,
# stop it first: `kubectl scale deployment/fraud-api --replicas=0`
docker-compose up -d

# 5. Run the MLOps pipeline components (in separate terminal windows)
python kafka_producer.py               # Stream 100 synthetic transactions
python kafka_consumer_v2.py            # Score transactions & log to predictions_log.csv
python -m streamlit run dashboard_v2.py # Launch the Human-in-the-Loop Analyst Workbench
python drift_monitor.py                # Run PSI drift check & auto-retrain trigger

Local URLs & Endpoints
Service
URL
Streamlit Analyst Dashboard - http://localhost:8501
API Documentation (Swagger) - http://localhost:8000/docs
API Health Check -  http://localhost:8000/health
Evidently Drift Report
reports/drift_report.html (Open locally in browser)

### AIML-01: Real-Time Medical Imaging and Diagnosis System

# 1. Python 3.11 venv (torch==2.4.1 has no cp313 wheels)
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Fill the wheelhouse for the Docker build (28 wheels, ~200 MB)
py -3.11 -m pip download -r requirements-api.txt -d wheelhouse \
    --find-links wheelhouse --only-binary=:all: \
    --platform manylinux2014_x86_64 --platform linux_x86_64 \
    --python-version 311 --implementation cp \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# 3. Copy the trained weights into artifacts/ (git-ignored, ~28 MB)
# Source: Kaggle kernel 03 output, Drive MyDrive/aiml02/artifacts/,
# or an older checkout of this repo.
cp <path>/densenet121_lungopacity.pt artifacts/

# 4. Start the three-container stack (Orthanc + API + ingest poller)
# On Windows, use API_PORT=8010 if port 8000 is held by Docker's own proxy.
# .env file: API_PORT=8010
docker compose up -d --build

# 5. Start the React viewer (separate terminal)
cd frontend
npm install
npm run dev

Local URLs
Service
URL
Radiologist Viewer - http://localhost:5173
API Docs (Swagger) - http://localhost:8000/docs (or :8010)
Health Check - http://localhost:8000/health
Orthanc UI - http://localhost:8042


📂 Repository Contents
AIML-01-Fraud-Platform/: Source code, K8s manifests, and Streamlit dashboard for the fraud system.
AIML-02-Medical-Imaging-Assist/: Source code, Docker configs, React frontend, and notebooks for the medical system.
Results Output Screenshots/: Visual proof of model training, drift monitoring, and inference results.

📄 License & Acknowledgments
Completed as part of the Nexvia Technologies AI/ML Internship Program (2026).
AIML-01 Data: IEEE-CIS Fraud Detection (Kaggle).
AIML-02 Data: RSNA Pneumonia Detection Challenge & Medical Segmentation Decathlon.
Disclaimer: AIML-02 is for educational purposes only. Not for clinical use.
