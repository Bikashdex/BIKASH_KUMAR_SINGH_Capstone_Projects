from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from datetime import timedelta

# 1. Define the Entity
card = Entity(name="card_id", join_keys=["card1"])

# 2. Define the Data Source (Pointing to your Parquet file)
fraud_source = FileSource(
    name="fraud_source",
    path=r"C:\Users\kumar\Desktop\AIML-01-Fraud-Platform\fraud_data_feast.parquet",
    timestamp_field="datetime",
)

# 3. Define the Feature View
transaction_features = FeatureView(
    name="transaction_features",
    entities=[card],
    ttl=timedelta(days=1),
    schema=[
        Field(name="TransactionAmt", dtype=Float32),
        Field(name="dist1", dtype=Float32),
    ],
    online=True,
    source=fraud_source,
)