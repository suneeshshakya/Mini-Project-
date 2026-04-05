import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ─── 1. Load Data ──────────────────────────────────────────
base_dir  = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Student_Mental_Health_Burnouts.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()
df = df.dropna()

print(f"✅ Loaded {len(df)} rows | Columns: {df.columns.tolist()}")

# ─── 2. Encode Gender ──────────────────────────────────────
le_gender = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
joblib.dump(le_gender, os.path.join(base_dir, "gender_encoder_v2.pkl"))

# ─── 3. Target Column ──────────────────────────────────────
# Use risk_level (Low / Medium / High) — already categorical
le_target = LabelEncoder()
y = le_target.fit_transform(df["risk_level"])
joblib.dump(le_target, os.path.join(base_dir, "label_encoder_v2.pkl"))

print(f"🎯 Classes: {le_target.classes_}")
for cls, idx in zip(le_target.classes_, range(len(le_target.classes_))):
    print(f"   {cls}: {sum(y == idx)} samples")

# ─── 4. Feature Selection ──────────────────────────────────
feature_cols = [
    "age", "gender", "academic_year",
    "study_hours_per_day", "exam_pressure", "academic_performance",
    "stress_level", "anxiety_score", "depression_score",
    "sleep_hours", "physical_activity", "social_support",
    "screen_time", "internet_usage", "financial_stress", "family_expectation"
]

X = df[feature_cols]
joblib.dump(feature_cols, os.path.join(base_dir, "feature_cols_v2.pkl"))

# ─── 5. Train/Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 6. Scaling ────────────────────────────────────────────
scaler   = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(base_dir, "scaler_v2.pkl"))

# ─── 7. Train & Compare Models ─────────────────────────────
models = {
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    "Random Forest":      RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

best_model, best_acc, best_name = None, 0, ""

print("\n🚀 Training Models...\n")
for name, mdl in models.items():
    mdl.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, mdl.predict(X_test_sc))
    print(f"  {name:25s} → Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc, best_model, best_name = acc, mdl, name

# ─── 8. Save Best Model ────────────────────────────────────
print(f"\n🏆 Best Model : {best_name}")
print(f"   Accuracy   : {best_acc:.4f}\n")

y_pred = best_model.predict(X_test_sc)
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

joblib.dump(best_model, os.path.join(base_dir, "dropout_model_v2.pkl"))
print("✅ Model saved as dropout_model_v2.pkl")

# ─── 9. Feature Importance ─────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    fi = sorted(
        zip(feature_cols, best_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n📊 Top Feature Importances:")
    for feat, imp in fi[:8]:
        bar = "█" * int(imp * 60)
        print(f"  {feat:25s} {bar} {imp:.3f}")
