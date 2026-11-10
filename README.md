# 🎓 Bikash Kumar Singh - Nexvia Technologies AI/ML Capstone Projects

**Intern ID:** NEX-INT-2026-0134  
**Batch:** October 2026  
**Specialization:** AI/Machine Learning

---

## 📁 Project Structure

This repository contains both capstone projects completed during the 30-day Nexvia Technologies AI/ML internship:

### 🔹 **AIML-01: Real-Time Financial Fraud Detection Platform**
End-to-end MLOps pipeline with Kafka streaming, XGBoost + GraphSAGE ensemble, Feast Feature Store, Kubernetes deployment, Evidently AI drift monitoring, and Streamlit human-in-the-loop dashboard.

**🔗 Live API:** https://aiml-01-fraud-platform.onrender.com  
**🔗 Live Dashboard:** https://aiml-01-fraud-platform.streamlit.app  
**📂 Code:** [`AIML-01/`](AIML-01/)

### 🔹 **AIML-02: AI-Assisted Medical Imaging and Diagnosis System**
Deep learning platform for X-ray/CT/MRI scan analysis with CNN models, Grad-CAM explainability heatmaps, and doctor review workflow.

**🔗 Live Demo:** [Your AIML-02 Live URL]  
**📂 Code:** [`AIML-02/`](AIML-02/)

---

## 📊 30-Day Curriculum Completion

✅ **Module 1:** Python, NumPy, Pandas, Data Cleaning & EDA  
✅ **Module 2:** Classical ML (Regression, Classification, Ensembles)  
✅ **Module 3:** Advanced ML & Deep Learning (XGBoost, PyTorch, GNN)  
✅ **Module 4:** MLOps (FastAPI, Git, Monitoring, Drift Detection)  
✅ **Module 5:** Capstone Projects (This Repository)

---

##  Quick Start

### AIML-01: Fraud Detection
```bash
cd AIML-01
pip install -r requirements.txt
docker-compose up -d
python kafka_producer.py
python -m streamlit run dashboard_v2.py