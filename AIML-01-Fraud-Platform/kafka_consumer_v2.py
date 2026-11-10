"""
Kafka Consumer v2: Reads transactions from Kafka, scores them via FastAPI,
AND logs every prediction to predictions_log.csv.
This log powers: drift monitoring (Evidently), the dashboard, and the
human-in-the-loop feedback cycle (analyst labels).
"""
from kafka import KafkaConsumer
import requests
import json
import csv
import os
from datetime import datetime

LOG_FILE = "predictions_log.csv"

# Create log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "transaction_id", "amount", "card_type",
            "fraud_probability", "ml_score", "rule_score",
            "risk_level", "is_fraud", "analyst_label"
        ])

consumer = KafkaConsumer(
    'fraud-transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='fraud-detection-group-v3'   # new group = re-reads old messages
)
print("✅ Kafka Consumer v2 connected (with prediction logging)!")
print("-" * 60)

API_URL = "http://localhost:8000/predict"
fraud_count = 0
total_count = 0

for message in consumer:
    transaction = message.value
    total_count += 1
    payload = {
        "transaction_id": transaction['transaction_id'],
        "amount": transaction['amount'],
        "card_type": transaction.get('product_cd', 'W')
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code != 200:
            print(f"❌ API ERROR {response.status_code}: {response.text}")
            continue
        result = response.json()

        # --- NEW: persist every prediction for monitoring & feedback ---
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                payload["transaction_id"],
                payload["amount"],
                payload["card_type"],
                result["fraud_probability"],
                result["ml_score"],
                result["rule_score"],
                result["risk_level"],
                int(result["is_fraud"]),
                ""   # analyst_label: filled later by the dashboard
            ])

        ts = datetime.now().strftime("%H:%M:%S")
        status = "🚨 FRAUD" if result.get('is_fraud') else "✅ SAFE"
        print(f"[{ts}] TXN {payload['transaction_id']} | ${payload['amount']:.2f} | {status} | Risk: {result['risk_level']} | Prob: {result['fraud_probability']:.4f}")
        if result.get('is_fraud'):
            fraud_count += 1
    except Exception as e:
        print(f"❌ Error processing {payload.get('transaction_id')}: {e}")

    if total_count >= 100:
        break

print("-" * 60)
print(f"✅ Done! Processed: {total_count} | Fraud flagged: {fraud_count}")
print(f"📁 All predictions saved to {LOG_FILE}")
consumer.close()