from feast import FeatureStore
import pandas as pd

# Initialize the store
store = FeatureStore(repo_path="./fraud_feature_repo")

# Fetch features for a sample card (e.g., card1 = 13000)
entity_df = pd.DataFrame({
    "card1": [13000],
    "datetime": ["2023-01-01 12:00:00"]
})

print("🔍 Fetching features from Feast...")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "transaction_features:TransactionAmt",
        "transaction_features:dist1",
    ]
).to_df()

print(training_df)
print("✅ Feast is working! Your API can now fetch these features.")