import pandas as pd

# 1. Read your existing CSV
print("Reading CSV...")
df = pd.read_csv('C:/Users/kumar/Desktop/AIML-01-Fraud-Platform/fraud_data_feast.csv')

# 2. Save it as a Parquet file (Industry Standard!)
print("Saving as Parquet...")
df.to_parquet('C:/Users/kumar/Desktop/AIML-01-Fraud-Platform/fraud_data_feast.parquet', index=False)

print("✅ Conversion Complete! You can now delete the CSV if you want.")