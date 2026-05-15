import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RAW_PATH  = "data/telco_churn.csv"
OUT_PATH  = "data/processed.csv"
MODEL_DIR = "models"

def load_data():
    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    print("Dataset not found – generating synthetic sample...")
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "customerID":       [f"C{i:04d}" for i in range(n)],
        "gender":           np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":    np.random.randint(0, 2, n),
        "Partner":          np.random.choice(["Yes", "No"], n),
        "Dependents":       np.random.choice(["Yes", "No"], n),
        "tenure":           np.random.randint(0, 72, n),
        "PhoneService":     np.random.choice(["Yes", "No"], n),
        "MultipleLines":    np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":  np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":   np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":     np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":      np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":      np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies":  np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":         np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod":    np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges":   np.round(np.random.uniform(20, 110, n), 2),
        "TotalCharges":     [str(round(t * m, 2)) for t, m in zip(np.random.randint(1, 72, n), np.random.uniform(20, 110, n))],
        "Churn":            np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    return df

def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    print(f"After cleaning: {df.shape[0]} rows")
    return df

def encode(df):
    df = df.copy()
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    print("Encoding done.")
    return df

def split_and_scale(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Train: {X_train_s.shape}  Test: {X_test_s.shape}")
    return X_train_s, X_test_s, y_train.values, y_test.values, X.columns.tolist()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    df = load_data()
    df = clean(df)
    df = encode(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    X_train, X_test, y_train, y_test, feature_names = split_and_scale(df)
    print("Preprocessing complete!")
