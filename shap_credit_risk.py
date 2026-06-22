"""
SHAP Credit Risk Analysis — Give Me Some Credit (Kaggle)
=========================================================
A complete ML pipeline demonstrating SHAP (SHapley Additive exPlanations)
for model interpretability on a real-world credit default dataset.

Dataset:  Give Me Some Credit — kaggle.com/c/GiveMeSomeCredit
          Download cs-training.csv and place it in the same directory.

Dependencies:
    pip install xgboost imbalanced-learn shap scikit-learn pandas numpy matplotlib

ROC-AUC achieved: ~0.816
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

np.random.seed(42)


# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────

df = pd.read_csv("cs-training.csv").drop(columns=["Unnamed: 0"])

df = df.rename(columns={
    "SeriousDlqin2yrs":                    "default",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio":                            "debt_ratio",
    "MonthlyIncome":                        "monthly_income",
    "NumberOfOpenCreditLinesAndLoans":      "open_credit_lines",
    "NumberOfTimes90DaysLate":              "late_90",
    "NumberRealEstateLoansOrLines":         "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents":                   "dependents",
})

# Impute missing values with median (safe for skewed financial data)
df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())
df["dependents"]     = df["dependents"].fillna(df["dependents"].median())

# Remove known data quality issues in this dataset
df = df[df["age"] > 0]
df = df[df["revolving_utilization"] <= 1.5]
df = df[df["debt_ratio"] <= 5]

print(f"Dataset: {df.shape[0]:,} rows | Default rate: {df['default'].mean():.2%}")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Debt-to-income: actual monthly debt burden
df["debt_to_income"] = (df["debt_ratio"] * df["monthly_income"]) / (df["monthly_income"] + 1)

# Total delinquencies across all severity buckets
df["total_late_payments"] = df["late_30_59"] + df["late_60_89"] + df["late_90"]

# Binary flag: utilization above 75% is a strong default signal
df["high_utilization"] = (df["revolving_utilization"] > 0.75).astype(int)

# Income per dependent: financial breathing room
df["income_per_dependent"] = df["monthly_income"] / (df["dependents"] + 1)

FEATURES = [
    "revolving_utilization", "age", "late_30_59", "debt_ratio",
    "monthly_income", "open_credit_lines", "late_90", "real_estate_loans",
    "late_60_89", "dependents",
    # engineered features
    "debt_to_income", "total_late_payments", "high_utilization", "income_per_dependent",
]

X = df[FEATURES]
y = df["default"]


# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")


# ─────────────────────────────────────────────
# 4. SMOTE — Handle Class Imbalance
# ─────────────────────────────────────────────
# ~6.9% default rate means the model will ignore the minority class
# without oversampling. SMOTE generates synthetic minority samples
# by interpolating between real ones. Applied to TRAINING SET ONLY.

print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")


# ─────────────────────────────────────────────
# 5. XGBOOST MODEL
# ─────────────────────────────────────────────

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_res, y_train_res, verbose=False)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nROC-AUC: {auc:.4f}")
print(classification_report(
    y_test, (y_pred_proba > 0.5).astype(int),
    target_names=["No Default", "Default"]
))


# ─────────────────────────────────────────────
# 6. SHAP — GLOBAL EXPLANATIONS
# ─────────────────────────────────────────────
# Global SHAP tells us: which features drive the model's decisions
# across ALL applicants? Think of it as an audit of the model's logic.

print("Computing SHAP values (this may take ~30s)...")
explainer = shap.TreeExplainer(model)
X_sample  = X_test.sample(2000, random_state=42)
shap_vals = explainer(X_sample)


def save(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── PLOT 1: Beeswarm — best global overview ──────────────────────────────
# Every dot = one applicant. Red = high feature value. Blue = low.
# X-axis = how much that feature pushed the default prediction up/down.
plt.figure(figsize=(11, 7))
shap.plots.beeswarm(shap_vals, max_display=14, show=False)
plt.title("Global SHAP — Feature Impact Across All Applicants", fontsize=13, pad=14)
save("shap_beeswarm.png")

# ── PLOT 2: Bar — clean feature importance ranking ────────────────────────
plt.figure(figsize=(9, 6))
shap.plots.bar(shap_vals, max_display=14, show=False)
plt.title("Global SHAP — Mean |SHAP Value| Feature Ranking", fontsize=13, pad=14)
save("shap_bar.png")


# ─────────────────────────────────────────────
# 7. SHAP — LOCAL EXPLANATIONS
# ─────────────────────────────────────────────
# Local SHAP tells us: why did the model make THIS specific prediction
# for THIS specific person? Essential for regulatory compliance and
# explaining rejections to applicants.

proba_sample = model.predict_proba(X_sample)[:, 1]
high_idx = proba_sample.argmax()   # worst-case applicant
low_idx  = proba_sample.argmin()   # best-case applicant

# ── PLOT 3: Waterfall — high-risk applicant ───────────────────────────────
# Starts at the baseline (avg prediction), adds each feature's contribution,
# arrives at this individual's predicted score.
plt.figure(figsize=(11, 7))
shap.plots.waterfall(shap_vals[high_idx], max_display=14, show=False)
plt.title("Local SHAP — Highest-Risk Applicant Explained", fontsize=13, pad=14)
save("shap_waterfall_high.png")

# ── PLOT 4: Waterfall — low-risk applicant ────────────────────────────────
plt.figure(figsize=(11, 7))
shap.plots.waterfall(shap_vals[low_idx], max_display=14, show=False)
plt.title("Local SHAP — Lowest-Risk Applicant Explained", fontsize=13, pad=14)
save("shap_waterfall_low.png")

# ── PLOT 5: Dependence — feature interaction ──────────────────────────────
# How does revolving utilization's SHAP value change with total late payments?
# Reveals interaction effects invisible to simple feature importance.
fig, ax = plt.subplots(figsize=(9, 5))
shap.dependence_plot(
    "revolving_utilization", shap_vals.values, X_sample,
    interaction_index="total_late_payments", ax=ax, show=False
)
ax.set_title("Dependence Plot — Utilization × Late Payments Interaction", fontsize=13)
save("shap_dependence.png")


# ─────────────────────────────────────────────
# 8. PROGRAMMATIC INSIGHTS
# ─────────────────────────────────────────────

shap_df    = pd.DataFrame(shap_vals.values, columns=FEATURES)
importance = shap_df.abs().mean().sort_values(ascending=False)

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (Mean |SHAP|)")
print("=" * 50)
for feat, val in importance.items():
    print(f"  {feat:<28} {val:.5f}")


def explain_applicant(idx: int, label: str = ""):
    """Print a human-readable breakdown of one applicant's prediction."""
    prob  = model.predict_proba(X_sample.iloc[[idx]])[0, 1]
    risk  = "HIGH RISK" if prob > 0.5 else "LOW RISK"
    sv    = shap_vals.values[idx]

    print(f"\n{'─'*55}")
    print(f"Applicant ({label}) → {risk} | Default Probability: {prob:.2%}")
    print(f"{'─'*55}")
    print(f"  {'Feature':<28} {'Value':>10}  {'SHAP':>8}  Direction")
    print(f"  {'-'*54}")

    for feat, val, s in sorted(
        zip(FEATURES, X_sample.iloc[idx], sv),
        key=lambda x: abs(x[2]), reverse=True
    ):
        arrow = "↑ more risk" if s > 0 else "↓ less risk"
        print(f"  {feat:<28} {val:>10.2f}  {s:>+8.4f}  {arrow}")


explain_applicant(high_idx, "Highest Risk")
explain_applicant(low_idx,  "Lowest Risk")

print(f"\n{'='*55}")
print(f"DONE — ROC-AUC: {auc:.4f} | All plots saved.")
print(f"{'='*55}")
