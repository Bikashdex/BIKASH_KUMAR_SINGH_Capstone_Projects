"""
Kafka Consumer: Reads transactions from Kafka and sends them to FastAPI for fraud prediction.
This simulates the real-time processing pipeline used in production banking systems.
"""

from kafka import KafkaConsumer
import requests
import json
from datetime import datetime

# 1. Connect to Kafka (NEW group_id so we re-read all messages from the start)
consumer = KafkaConsumer(
    'fraud-transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='fraud-detection-group-v2'
)

print("✅ Kafka Consumer connected!")
print("👂 Listening for transactions on 'fraud-transactions' topic...")
print("-" * 60)

# 2. FastAPI endpoint
API_URL = "http://localhost:8000/predict"

# 3. Process each transaction
fraud_count = 0
total_count = 0

for message in consumer:
    transaction = message.value
    total_count += 1

    # Prepare payload for FastAPI
    payload = {
        "transaction_id": transaction['transaction_id'],
        "amount": transaction['amount'],
        "card_type": transaction.get('product_cd', 'W')
    }

    try:
        # Send to FastAPI for prediction
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code != 200:
            print(f"❌ API ERROR {response.status_code}: {response.text}")
            continue

        result = response.json()

        # Display results
        timestamp = datetime.now().strftime("%H:%M:%S")
        fraud_status = "🚨 FRAUD" if result.get('is_fraud') else "✅ SAFE"
        risk = result.get('risk_level', 'UNKNOWN')
        probability = result.get('fraud_probability', 0)

        print(f"[{timestamp}] TXN {transaction['transaction_id']} | ${transaction['amount']:.2f} | {fraud_status} | Risk: {risk} | Prob: {probability:.4f}")

        if result.get('is_fraud'):
            fraud_count += 1

    except Exception as e:
        print(f"❌ Error processing transaction {transaction['transaction_id']}: {e}")

    # Stop after processing 100 transactions (for demo)
    if total_count >= 100:
        break

print("-" * 60)
print(f"\n✅ Processing Complete!")
print(f"Total transactions processed: {total_count}")
print(f"Fraud detected: {fraud_count}")
print(f"Fraud rate: {(fraud_count/total_count*100):.2f}%")

consumer.close()