from feast import FeatureStore
import pandas as pd

# Initialize the store
store = FeatureStore(repo_path="./fraud_feature_repo")

# Pick a real card ID from your parquet file
df = pd.read_parquet(
    r"C:\Users\kumar\Desktop\AIML-01-Fraud-Platform\fraud_data_feast.parquet",
    columns=["card1"],
)
sample_card = int(df["card1"].iloc[0])
print(f"🔍 Fetching online features for card1={sample_card} ...")

# Fetch from the ONLINE store (same call your FastAPI will make)
result = store.get_online_features(
    features=[
        "transaction_features:TransactionAmt",
        "transaction_features:dist1",
    ],
    entity_rows=[{"card1": sample_card}],
)

print(pd.DataFrame(result.to_dict()))
print("✅ Feast is working! Your API can now fetch these features.")