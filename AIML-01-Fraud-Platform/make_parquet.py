import pandas as pd

print("🔄 Loading CSV data (this might take 30 seconds)...")
# Load your main data
df = pd.read_csv('train_transaction.csv')

print("⏳ Adding 'datetime' column (Feast needs this)...")
# Feast requires a datetime column. We create one based on TransactionDT (seconds)
df['datetime'] = pd.to_datetime(df['TransactionDT'], unit='s', origin='2023-01-01')

print("💾 Saving as Parquet file...")
# Save it with the exact name Feast is looking for
df.to_parquet('fraud_data_feast.parquet', index=False)

print("✅ SUCCESS! 'fraud_data_feast.parquet' has been created.")