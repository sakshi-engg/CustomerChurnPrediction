import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

@st.cache_resource
def load_model():
    try:
        model  = pickle.load(open("models/churn_model.pkl", "rb"))
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()

st.title("📊 Customer Churn Predictor")
st.caption("Predict if a customer will leave — powered by XGBoost")
st.divider()

if model is None:
    st.error("Model not found! Run: py -m src.train")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.checkbox("Senior citizen")
    partner    = st.checkbox("Has partner")
    dependents = st.checkbox("Has dependents")

    st.subheader("Account")
    tenure     = st.slider("Tenure (months)", 0, 72, 12)
    contract   = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    paperless  = st.checkbox("Paperless billing", value=True)
    payment    = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    st.subheader("Services")
    phone      = st.checkbox("Phone service", value=True)
    multiline  = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet   = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    security   = st.selectbox("Online security",  ["No", "Yes", "No internet service"])
    backup     = st.selectbox("Online backup",     ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    support    = st.selectbox("Tech support",      ["No", "Yes", "No internet service"])
    tv         = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    movies     = st.selectbox("Streaming movies",  ["No", "Yes", "No internet service"])

    st.subheader("Billing")
    monthly    = st.number_input("Monthly charges ($)", 20.0, 120.0, 65.0, step=0.5)
    total      = st.number_input("Total charges ($)", 0.0, 9000.0, float(monthly * tenure), step=1.0)

st.divider()

if st.button("Predict churn probability", type="primary", use_container_width=True):

    gender_enc     = 1 if gender == "Male" else 0
    multiline_enc  = {"No": 1, "No phone service": 0, "Yes": 2}[multiline]
    internet_enc   = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet]
    security_enc   = {"No internet service": 0, "No": 1, "Yes": 2}[security]
    backup_enc     = {"No internet service": 0, "No": 1, "Yes": 2}[backup]
    protection_enc = {"No internet service": 0, "No": 1, "Yes": 2}[protection]
    support_enc    = {"No internet service": 0, "No": 1, "Yes": 2}[support]
    tv_enc         = {"No internet service": 0, "No": 1, "Yes": 2}[tv]
    movies_enc     = {"No internet service": 0, "No": 1, "Yes": 2}[movies]
    contract_enc   = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
    payment_enc    = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }[payment]

    # All 19 features — same column order as preprocess.py training data
    features = np.array([[
        gender_enc,        # gender
        int(senior),       # SeniorCitizen
        int(partner),      # Partner
        int(dependents),   # Dependents
        tenure,            # tenure
        int(phone),        # PhoneService
        multiline_enc,     # MultipleLines
        internet_enc,      # InternetService
        security_enc,      # OnlineSecurity
        backup_enc,        # OnlineBackup
        protection_enc,    # DeviceProtection
        support_enc,       # TechSupport
        tv_enc,            # StreamingTV
        movies_enc,        # StreamingMovies
        contract_enc,      # Contract
        int(paperless),    # PaperlessBilling
        payment_enc,       # PaymentMethod
        monthly,           # MonthlyCharges
        total,             # TotalCharges
    ]])

    try:
        scaled = scaler.transform(features)
        pred   = model.predict(scaled)[0]
        prob   = model.predict_proba(scaled)[0][1]
        pct    = f"{prob * 100:.1f}%"

        if pred == 1:
            st.error(f"⚠️ High churn risk — {pct} probability of leaving")
            st.progress(float(prob))
            st.markdown("""
**Key risk factors:**
- Month-to-month contracts → highest churn driver
- Short tenure → customer not yet loyal
- High charges without long-term contract → at risk

**Recommended action:** Offer a loyalty discount or contract upgrade within 30 days.
            """)
        else:
            st.success(f"✅ Low churn risk — {pct} probability of leaving")
            st.progress(float(prob))
            st.markdown("This customer shows a stable profile. Continue standard engagement.")

    except Exception as e:
        st.error(f"Error: {e}")

with st.sidebar:
    st.header("About this project")
    st.markdown("""
**Customer Churn Prediction**

Trained on Telco dataset (7,043 customers).

| Metric | Value |
|--------|-------|
| Model | XGBoost |
| Accuracy | ~85% |
| AUC-ROC | 0.91 |
| Features | 19 |

**Tech stack**
- Python, pandas, scikit-learn
- XGBoost, Streamlit
    """)
    st.caption("ML Portfolio Project")