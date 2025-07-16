import pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load features and drop metadata cols
df = pd.read_csv("data/features.csv")
spoof = df["label"]
feats = df.drop(columns = ["file", "label"]).astype("float32")

# Split into training-test sets (80-20)
feats_train, feats_test, spoof_train, spoof_test = train_test_split(feats, 
                        spoof, test_size = 0.2, random_state = 10)

# Scale features for better logreg convergence
scaler = StandardScaler().fit(feats_train)
feats_train_scaled = scaler.transform(feats_train)
feats_test_scaled = scaler.transform(feats_test)

# Hyperparam tuning to get best regularisation strength (C)
param_grid = {"C": [0.01, 0.1, 1]}
grid = GridSearchCV(LogisticRegression(max_iter = 5000, random_state = 10), 
                    param_grid, scoring = "roc_auc", cv = 5, n_jobs = 4)
grid.fit(feats_train_scaled, spoof_train)
best_C = grid.best_params_["C"]
print(f"Best C = {best_C} --- mean ROC AUC = {grid.best_score_:.4f}")

# Refit model with best C
model = LogisticRegression(max_iter = 5000, C = best_C)
model.fit(feats_train_scaled, spoof_train)

# Evaluate with roc-auc and avg precision
probs = model.predict_proba(feats_test_scaled)[:, 1]
roc_auc = roc_auc_score(spoof_test, probs)
avg_prec = average_precision_score(spoof_test, probs)
print(f"Test ROC AUC: {roc_auc:.4f} --- Test avg precision: {avg_prec:.4f}")
print(classification_report(spoof_test, model.predict(feats_test_scaled), 
                            target_names = ["bonafide", "spoof"]))

# Save model and write results
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(model, "models/logreg_best_C.joblib")

with open("results/baseline_metrics.txt", "w") as f:
    f.write(f"CV best C: {best_C}\n")
    f.write(f"CV mean ROC AUC: {grid.best_score_:.4f}\n")
    f.write(f"Final ROC AUC: {roc_auc:.4f}\n")