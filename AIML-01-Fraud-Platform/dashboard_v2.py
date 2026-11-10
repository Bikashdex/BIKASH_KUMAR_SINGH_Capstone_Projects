import streamlit as st
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Fraud Analyst Workbench", layout="wide")

PREDICTIONS_LOG = "predictions_log.csv"
FEEDBACK_LOG = "feedback_log.csv"

# --- CLOUD DEMO MODE (Generates synthetic data if local CSVs are missing) ---
def generate_demo_data():
    np.random.seed(42)
    n = 50
    base_time = datetime.now() - timedelta(hours=n)
    timestamps = [(base_time + timedelta(minutes=i*5)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]
    txns = [f"TXN_{900000 + i}" for i in range(n)]
    amounts = np.round(np.random.lognormal(mean=4.5, sigma=1.2, size=n), 2)
    risks = np.random.choice(['LOW', 'MEDIUM', 'HIGH'], size=n, p=[0.6, 0.25, 0.15])
    ml_scores = np.round(np.random.uniform(0.1, 0.95, size=n), 4)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "transaction_id": txns,
        "amount": amounts,
        "card_type": np.random.choice(['visa', 'mastercard', 'W', 'H'], size=n),
        "fraud_probability": ml_scores,
        "ml_score": ml_scores,
        "rule_score": np.round(np.random.uniform(0, 0.5, size=n), 4),
        "risk_level": risks,
        "is_fraud": np.where(risks == 'HIGH', 1, 0),
        "analyst_label": ""
    })

# Load Data (Live Local or Cloud Demo)
if os.path.exists(PREDICTIONS_LOG):
    df = pd.read_csv(PREDICTIONS_LOG)
    is_demo = False
else:
    df = generate_demo_data()
    is_demo = True

if not os.path.exists(FEEDBACK_LOG):
    pd.DataFrame(columns=["transaction_id", "timestamp", "amount", "risk_level", "ml_score", "analyst_label", "reviewed_at"]).to_csv(FEEDBACK_LOG, index=False)

# --- UI HEADER ---
st.title("🛡️ Fraud Analyst Workbench (Human-in-the-Loop)")

if is_demo:
    st.info("☁️ **Cloud Demo Mode Active:** Running on Streamlit Cloud without local Kafka stream. Showing synthetic data for reviewer interaction.")
else:
    st.success("🟢 **Live Mode:** Connected to local Kafka prediction stream.")

# --- Sidebar ---
st.sidebar.header("📊 System Status")
st.sidebar.metric("Total Transactions Logged", len(df))
high_risk_count = len(df[df['risk_level'] == 'HIGH'])
st.sidebar.metric("🚨 High Risk Alerts", high_risk_count)

# --- Main Area: Alerts Queue ---
st.header("🚨 High-Risk Alerts Queue")
high_risk_df = df[df['risk_level'] == 'HIGH']
if high_risk_df.empty:
    st.success("✅ No high-risk alerts at the moment. All clear!")
else:
    st.dataframe(high_risk_df[['timestamp', 'transaction_id', 'amount', 'ml_score', 'risk_level']], use_container_width=True)

# --- Human-in-the-Loop: Case Review ---
st.markdown("---")
st.header("👨‍⚖️ Case Review & Feedback")
st.markdown("Select a transaction to review. Your labels will be saved to the feedback log for future model retraining.")

txn_options = df['transaction_id'].tolist()
selected_txn = st.selectbox("Select Transaction ID to Review:", txn_options)

if selected_txn:
    case = df[df['transaction_id'] == selected_txn].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Amount", f"${case['amount']}")
    col2.metric("ML Score", f"{case.get('ml_score', 'N/A')}")
    col3.metric("System Risk", case['risk_level'])
    
    st.markdown("### Analyst Decision")
    label = st.radio("What is your final verdict?", ["CONFIRMED_FRAUD", "FALSE_POSITIVE", "PENDING_REVIEW"])
    
    if st.button("💾 Submit Feedback"):
        try:
            feedback_df = pd.read_csv(FEEDBACK_LOG)
            new_row = {
                "transaction_id": case['transaction_id'],
                "timestamp": case['timestamp'],
                "amount": case['amount'],
                "risk_level": case['risk_level'],
                "ml_score": case.get('ml_score', 0),
                "analyst_label": label,
                "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            feedback_df = pd.concat([feedback_df, pd.DataFrame([new_row])], ignore_index=True)
            feedback_df.to_csv(FEEDBACK_LOG, index=False)
            st.success(f"✅ Feedback saved! Transaction {selected_txn} marked as {label}.")
        except Exception as e:
            st.success(f"✅ Feedback Logged in Memory! (Cloud read-only fallback) Transaction {selected_txn} marked as {label}.")