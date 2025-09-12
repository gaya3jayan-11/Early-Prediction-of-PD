# scripts/model_training.py

import os, json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint, uniform

# ===============================
# Step 1: Load dataset (CSV)
# ===============================
print("🔹 Loading dataset...")
df = pd.read_csv("data/processed/final_dataset_enhanced.csv")
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===============================
# Step 2: Features and labels (+ leakage guard)
# ===============================
y = df["convert_84m"]

# start from all columns except the label
X = df.drop(columns=["convert_84m"], errors="ignore")

# drop obvious identifiers / dates / potential leakage
LEAKY_PREFIXES = ("convert", "diagnos", "dx", "parkinson", "pd_", "event", "status", "group", "cohort", "follow", "case", "control")
leaky = [c for c in X.columns if c.lower().startswith(LEAKY_PREFIXES)]
extra_ids = [c for c in ["PATNO", "subject_id", "baseline_date_final"] if c in X.columns]
drop_leak = list(set(leaky + extra_ids))
if drop_leak:
    print(f"⚠️ Dropping potential leakage columns: {drop_leak}")
    X = X.drop(columns=drop_leak, errors="ignore")

print(f"✅ Initial features shape: {X.shape}, Labels shape: {y.shape}")

# ===============================
# Step 3: Clean useless features
# ===============================
nan_cols = X.columns[X.isna().all()]
const_cols = X.columns[X.nunique() <= 1]
drop_cols = set(nan_cols) | set(const_cols)

if drop_cols:
    print(f"⚠️ Dropping useless columns: {list(drop_cols)}")
    X = X.drop(columns=drop_cols)

print(f"✅ Cleaned features shape: {X.shape}")

# Save the final feature list (used later for inference)
feature_names = list(X.columns)

# ===============================
# Step 4: Train/test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# ===============================
# Step 5: Scaling (fit once here; pipeline uses scaled data)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# ===============================
# Step 6: Models + pipelines
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        random_state=42, scale_pos_weight=1, use_label_encoder=False, eval_metric="logloss"
    )
}

# imputer + SMOTE + model (we feed in scaled arrays)
pipelines = {
    name: ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])
    for name, model in models.items()
}

# ===============================
# Step 7: Cross-validation (on scaled train)
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{name} Mean ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# ===============================
# Step 8: Directories
# ===============================
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ===============================
# Step 9: Hyperparameter tuning
# ===============================
param_grids = {
    "Random Forest": {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(3, 20),
        "model__min_samples_split": randint(2, 10),
        "model__min_samples_leaf": randint(1, 5)
    },
    "SVM": {
        "model__C": uniform(0.1, 10),
        "model__gamma": ["scale", "auto"]
    },
    "XGBoost": {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4)
    }
}

best_models = {}
for name in ["Random Forest", "SVM", "XGBoost"]:
    print(f"\n🔹 Tuning {name}...")
    pipeline = pipelines[name]
    param_dist = param_grids[name]

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=20,
        scoring="roc_auc", cv=5, n_jobs=-1, verbose=1, random_state=42
    )
    search.fit(X_train_scaled, y_train)
    best_models[name] = search.best_estimator_
    print(f"✅ {name} best params: {search.best_params_}")
    print(f"   Best CV ROC-AUC: {search.best_score_:.3f}")

# ===============================
# Step 10: Select best model
# ===============================
best_name = max(
    best_models,
    key=lambda k: cross_val_score(best_models[k], X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()
)
best_pipeline = best_models[best_name]
print(f"\n🏆 Best model: {best_name}")

# ===============================
# Step 11: Save pipeline + artifacts
# ===============================
joblib.dump(best_pipeline, "models/Best_Model_pipeline.pkl")
joblib.dump(scaler, "models/scaler.pkl")
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("✅ Best pipeline saved: models/Best_Model_pipeline.pkl")
print("✅ Scaler saved: models/scaler.pkl")
print("✅ Feature list saved: models/feature_names.json")

# ===============================
# Step 12: Test set evaluation & plots
# ===============================
y_proba = best_pipeline.predict_proba(X_test_scaled)[:, 1]
y_pred = best_pipeline.predict(X_test_scaled)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{best_name} ROC Curve')
plt.legend()
plt.savefig(f"plots/{best_name.replace(' ', '_')}_ROC_curve.png")
plt.close()

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'{best_name} Precision-Recall Curve')
plt.legend()
plt.savefig(f"plots/{best_name.replace(' ', '_')}_PR_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'{best_name} Confusion Matrix')
plt.savefig(f"plots/{best_name.replace(' ', '_')}_confusion_matrix.png")
plt.close()

# ===============================
# Step 13: Feature importance (tree models)
# ===============================
if best_name in ["Random Forest", "XGBoost"]:
    model = best_pipeline.named_steps['model']
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    top_idx = sorted_idx[:20]
    top_features = np.array(feature_names)[top_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], importances[top_idx][::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"{best_name} Top 20 Features")
    plt.tight_layout()
    plt.savefig(f"plots/{best_name.replace(' ', '_')}_top_features.png")
    plt.close()
    print("✅ Feature importance plot saved")

# ===============================
# Step 14: Prediction helper (uses saved scaler + feature list)
# ===============================
def predict_new_data(csv_path: str, output_path: str = "models/predictions.csv"):
    """
    Load new CSV, align columns to training features, scale with saved scaler,
    apply trained pipeline, and save predictions + probabilities.
    """
    X_new_raw = pd.read_csv(csv_path)

    # Load artifacts
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_names.json", "r") as f:
        feats = json.load(f)

    # Align to training columns (missing -> NaN; extra columns ignored)
    X_new = X_new_raw.reindex(columns=feats, fill_value=np.nan)

    # Scale then predict
    X_new_scaled = scaler.transform(X_new)
    y_pred = best_pipeline.predict(X_new_scaled)
    y_proba = best_pipeline.predict_proba(X_new_scaled)[:, 1]

    out = X_new_raw.copy()
    out["prediction"] = y_pred
    out["probability"] = y_proba
    out.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
