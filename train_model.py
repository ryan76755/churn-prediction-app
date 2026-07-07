import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)
import joblib
import json

df = pd.read_csv("telco_churn.csv")

# Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df.drop('customerID', axis=1, inplace=True)

target = 'Churn'
df[target] = df[target].map({'Yes': 1, 'No': 0})

cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in df.columns if c not in cat_cols and c != target]

encoders = {}
df_enc = df.copy()
for c in cat_cols:
    le = LabelEncoder()
    df_enc[c] = le.fit_transform(df_enc[c])
    encoders[c] = le

X = df_enc.drop(target, axis=1)
y = df_enc[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)
}

results = {}
best_model = None
best_f1 = -1
best_name = None

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4)
    }
    if results[name]["f1"] > best_f1:
        best_f1 = results[name]["f1"]
        best_model = model
        best_name = name

print(json.dumps(results, indent=2))
print("Best model:", best_name)

# Feature importance (Random Forest)
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Feature Importances:\n", importances.head(10))

# Save artifacts
joblib.dump(rf, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

with open("model_results.json", "w") as f:
    json.dump({
        "results": results,
        "best_model": best_name,
        "feature_importance": importances.head(10).to_dict(),
        "cat_cols": cat_cols,
        "num_cols": num_cols
    }, f, indent=2)

print("\nArtifacts saved.")
