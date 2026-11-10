"""
Auto-Retrain: sklearn Pipeline + 3-fold Stratified CV + RandomizedSearchCV.
"""
import pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, average_precision_score
from xgboost import XGBClassifier

print("Loading training data (RAM-safe sample)...")
df = pd.read_csv('train_transaction.csv').sample(n=100000, random_state=42)

# Same cleaning rules as original training (consistency = no skew)
missing = df.isnull().mean()
df = df.drop(columns=missing[missing > 0.8].index)
num = df.select_dtypes(include=['float64', 'int64']).columns
df[num] = df[num].fillna(0)
for c in df.select_dtypes(include=['object']).columns:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'], errors='ignore')
y = df['isFraud']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([("clf", XGBClassifier(tree_method='hist', random_state=42, eval_metric='logloss'))])
grid = {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1, 0.2], "clf__scale_pos_weight": [10, 27, 40]}

print("Running 3-fold CV hyperparameter search (few minutes)...")
search = RandomizedSearchCV(pipe, grid, n_iter=6, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                            scoring=make_scorer(average_precision_score), random_state=42, n_jobs=-1, verbose=1)
search.fit(X_tr, y_tr)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV PR-AUC: {search.best_score_:.4f}")
proba = search.best_estimator_.predict_proba(X_te)[:, 1]
print(f"Holdout PR-AUC: {average_precision_score(y_te, proba):.4f}")
joblib.dump(search.best_estimator_, "fraud_detection_model_v2.pkl")
print("✅ Saved fraud_detection_model_v2.pkl — rebuild Docker to serve it.")