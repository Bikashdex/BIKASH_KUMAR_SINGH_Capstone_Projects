import pandas as pd
from datetime import datetime, timedelta

# Load your cleaned data (make sure train_transaction.csv is in your main folder)
df = pd.read_csv('train_transaction.csv')

# Create a dummy datetime column (Feast requires this)
start_date = datetime(2023, 1, 1)
df['datetime'] = df['TransactionDT'].apply(lambda x: start_date + timedelta(seconds=x))

# Save a Feast-ready version
df.to_csv('fraud_data_feast.csv', index=False)
print("✅ Data prepared for Feast!")