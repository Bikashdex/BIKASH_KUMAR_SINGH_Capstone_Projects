"""
Kafka Producer: Simulates real-time transaction streaming
This script reads from the dataset and sends transactions to Kafka one by one,
mimicking how a real payment gateway would stream live transactions.
"""

from kafka import KafkaProducer
import pandas as pd
import json
import time
import random

# 1. Connect to Kafka (running in Docker)
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("✅ Connected to Kafka!")
print(" Starting transaction stream...")

# 2. Load a sample of transactions to "replay"
# In production, this would be live data from a payment gateway
# Use raw string (r"") for Windows paths to avoid backslash errors
df = pd.read_csv('train_transaction.csv')  # Adjust path if needed
sample_df = df.head(100)  # Just stream 100 transactions for demo

# 3. Stream transactions to Kafka
topic_name = 'fraud-transactions'

for index, row in sample_df.iterrows():
    # Create a transaction payload
    transaction = {
        'transaction_id': str(row['TransactionID']),
        'amount': float(row['TransactionAmt']),
        'product_cd': str(row.get('ProductCD', 'W')),
        'card1': int(row.get('card1', 0)),
        'is_fraud_actual': int(row['isFraud'])  # For validation later
    }
    
    # Send to Kafka
    producer.send(topic_name, value=transaction)
    print(f" Sent Transaction {transaction['transaction_id']} | Amount: ${transaction['amount']}")
    
    # Simulate real-time delay (1 transaction per second)
    time.sleep(1)

print("\n✅ Finished streaming 100 transactions to Kafka!")
producer.flush()
producer.close()