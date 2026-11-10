from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()

def add_slide(title, content):
    slide_layout = prs.slide_layouts[1] # Bullet layout
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.text = content[0]
    for item in content[1:]:
        p = tf.add_paragraph()
        p.text = item

# Slide 1: Title
slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(slide_layout)
slide.shapes.title.text = "Real-Time Financial Fraud Detection Platform"
slide.placeholders[1].text = "An End-to-End MLOps Pipeline with Graph Neural Networks\nBikash Kumar Singh | AI/ML Engineering Intern | Nexvia Technologies"

# Slides 2-10
add_slide("The Business Problem", [
    "Fraud costs the industry billions annually.",
    "The Data Reality: Highly imbalanced datasets (only ~3.5% fraud).",
    "The Trap: Standard 'Accuracy' is a misleading metric.",
    "The Goal: Optimize for PR-AUC, operate in real-time (<100ms), and prevent silent model decay."
])

add_slide("High-Level Architecture", [
    "Offline Training: Colab + MLflow + Feast Feature Store",
    "Real-Time Serving: Kafka → FastAPI → Kubernetes (2 Replicas)",
    "Monitoring: Evidently AI + PSI Drift Triggers",
    "Feedback: Streamlit Human-in-the-Loop Dashboard",
    "[INSERT YOUR ARCHITECTURE DIAGRAM PNG HERE]"
])

add_slide("Data & Advanced Feature Engineering", [
    "Raw Data: 590,000+ transactions, 300+ anonymized V-features.",
    "Engineered Signals: Velocity (time since last txn), Z-Score (amount deviation), Device/Location Shift.",
    "Impact: Gives the model 'context' rather than just raw numbers."
])

add_slide("The ML Core: Ensemble Modeling", [
    "Baselines: Logistic Regression (0.188 PR-AUC) vs Random Forest (0.595).",
    "The Champion: XGBoost with scale_pos_weight=27 to handle class imbalance.",
    "The Innovation: GraphSAGE GNN (PyTorch Geometric) to detect fraud rings.",
    "The Result: Ensemble blend (80% XGB + 20% GNN) achieved a PR-AUC of 0.615."
])

add_slide("MLOps & The Feature Store", [
    "The Silent Killer: Training-Serving Skew.",
    "The Solution: Feast (Feature Store) - Offline Parquet + Online SQLite.",
    "Serving: Dockerized FastAPI microservice with Pydantic validation, deployed on Kubernetes."
])

add_slide("Monitoring & Automated Retraining", [
    "The Reality: Models decay the moment they hit production (Data Drift).",
    "The Metric: PSI (Population Stability Index) — the banking industry standard.",
    "< 0.10: Stable | > 0.20: 🚨 Trigger Retraining",
    "The Loop: drift_monitor.py auto-launches retrain.py (Sklearn Pipelines + Stratified K-Fold CV)."
])

add_slide("Responsible AI & Human-in-the-Loop", [
    "Explainability (SHAP): Proves the model relies on logical features, not biased proxies.",
    "Fairness Audit: Evaluated PR-AUC across card4 subgroups.",
    "The Workbench: Streamlit dashboard queues high-risk alerts for analyst review.",
    "Analysts label cases (CONFIRMED_FRAUD / FALSE_POSITIVE) for next week's retraining."
])

add_slide("Business Value & ROI", [
    "⚡ Speed: Kafka + K8s ensures sub-100ms transaction scoring.",
    "📈 Scale: Kubernetes auto-scales and self-heals if pods crash.",
    "🛡️ Safety: Automated drift detection prevents silent financial losses.",
    "Conclusion: Transitioned from a static Jupyter Notebook to a resilient, enterprise-grade MLOps platform."
])

add_slide("Curriculum Mapping & Conclusion", [
    "Module 1 & 2: Data prep, EDA, Baseline comparisons.",
    "Module 3: Advanced ML, XGBoost, PyTorch Geometric (GNN).",
    "Module 4: FastAPI, Feast, Evidently Drift, SHAP.",
    "Module 5: End-to-end deployment, Git hygiene, API documentation.",
    "Final Thought: 'This project isn't just code; it's the synthesis of 30 days of rigorous AI/ML engineering.'"
])

prs.save('AIML-01_Presentation.pptx')
print("✅ Deck saved as AIML-01_Presentation.pptx!")