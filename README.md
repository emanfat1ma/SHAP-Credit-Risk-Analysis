# SHAP Credit Risk Analysis

An end-to-end machine learning project that builds a credit default prediction model and explains its decisions using SHAP (SHapley Additive exPlanations).

## Overview

This project trains an XGBoost classifier on the Kaggle "Give Me Some Credit" dataset to predict loan default risk. Beyond prediction, it focuses on **interpretability**, showing why a model makes specific decisions using SHAP.

## What This Project Does

* Cleans and preprocesses credit risk data
* Engineers financial risk features
* Handles class imbalance using SMOTE
* Trains an XGBoost model (~0.81 ROC-AUC)
* Applies SHAP for model explainability
* Generates global + local interpretation plots
* Provides human-readable explanations per applicant

## Key Insight

Instead of treating the model as a black box, this project explains:

* Which features matter most overall
* Why a specific applicant is high/low risk
* How individual financial factors influence predictions

## Explainability Outputs

* SHAP Beeswarm → global feature impact
* SHAP Bar Plot → feature importance ranking
* SHAP Waterfall → individual predictions
* SHAP Dependence Plot → feature interactions

## Tech Stack

Python · XGBoost · SHAP · Scikit-learn · Pandas · NumPy · Matplotlib · Imbalanced-learn

## Dataset

Kaggle: Give Me Some Credit
(Requires downloading `cs-training.csv` separately)

## Result

* ROC-AUC: ~0.816
* Fully interpretable predictions using SHAP

---

Built for learning, interpretability, and real-world ML thinking.
