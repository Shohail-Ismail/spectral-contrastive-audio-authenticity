import pandas as pd, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Load features and drop metadata cols
df = pd.read_csv("data/features.csv")
spoof = df["label"]
feats = df.drop(columns = ["file", "label"])

# Split (80-20)
cut = int(len(feats) * 0.8)
feats_train = feats.iloc[:cut]
spoof_train = spoof.iloc[:cut]
feats_test = feats.iloc[cut:]
spoof_test = spoof.iloc[cut:]

# Train model
model = LogisticRegression(max_iter = 500)
model.fit(feats_train, spoof_train)

# Evaluate with roc-auc and the usual suspects
probs = model.predict_proba(feats_test)[:, 1]
roc_auc = roc_auc_score(spoof_test, probs)
pred_labels = model.predict(feats_test)
report = classification_report(spoof_test, pred_labels, target_names=["bonafide", "spoof"])
print(f"ROC AUC: {roc_auc:.4f}")
print(report)

# Save model and write results
model_path = "models/logreg_mk1.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open("results/baseline_metrics.txt", "w") as f:
    f.write(f"ROC AUC: {roc_auc:.4f}")
