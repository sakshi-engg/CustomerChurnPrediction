import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not found. Install with: pip install xgboost")

MODEL_DIR = "models"

def load_and_prepare():
    from src.preprocess import load_data, clean, encode, split_and_scale
    df = load_data()
    df = clean(df)
    df = encode(df)
    return split_and_scale(df)

def evaluate(model, X_test, y_test, label=""):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report.get("1", {}).get("f1-score", 0)
    print(f"\n{label} — Accuracy: {acc:.3f}  AUC: {auc:.3f}  F1(churn): {f1:.3f}")
    return {"accuracy": acc, "auc": auc, "f1_churn": f1}

def plot_feature_importance(model, name, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    os.makedirs("notebooks", exist_ok=True)
    imp = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": imp})
    feat_df = feat_df.sort_values("importance", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df["feature"], feat_df["importance"], color="#534AB7")
    ax.set_title(f"Feature importance – {name}")
    plt.tight_layout()
    plt.savefig("notebooks/feature_importance.png", dpi=150)
    print("Saved: notebooks/feature_importance.png")
    plt.close()

def plot_roc(models_dict, X_test, y_test):
    os.makedirs("notebooks", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Logistic Regression": "#888780", "Random Forest": "#534AB7", "XGBoost": "#0F6E56"}
    for label, model in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})", color=colors.get(label, "gray"), lw=1.8)
    ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Model comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("notebooks/roc_curve.png", dpi=150)
    print("Saved: notebooks/roc_curve.png")
    plt.close()

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    results = {}
    trained_models = {}

    print("\n── Training Logistic Regression ──")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = evaluate(lr, X_test, y_test, "Logistic Regression")
    trained_models["Logistic Regression"] = lr

    print("\n── Training Random Forest ──")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate(rf, X_test, y_test, "Random Forest")
    trained_models["Random Forest"] = rf

    if XGB_AVAILABLE:
        print("\n── Training XGBoost ──")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, eval_metric="logloss", random_state=42, verbosity=0
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        results["XGBoost"] = evaluate(xgb_model, X_test, y_test, "XGBoost")
        trained_models["XGBoost"] = xgb_model

    print("\n\n══ Final Results ══════════════════════")
    print(f"{'Model':<22} {'Accuracy':>10} {'AUC-ROC':>10}")
    print("─" * 44)
    for name, res in results.items():
        print(f"{name:<22} {res['accuracy']:>10.3f} {res['auc']:>10.3f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/churn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to models/churn_model.pkl")

    plot_feature_importance(best_model, best_name, feature_names)
    plot_roc(trained_models, X_test, y_test)
    print("\nTraining complete!")
