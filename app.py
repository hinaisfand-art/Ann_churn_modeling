from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Salary prediction",
    page_icon=":material/payments:",
)

st.title("Bank customer salary prediction")
st.caption("Learning project: predicts EstimatedSalary from customer details.")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")
    model = tf.keras.models.load_model(BASE_DIR / "salary_model.keras")
    return preprocessor, model


try:
    preprocessor, model = load_artifacts()
except Exception as error:
    st.error(f"Could not load the saved model files: {error}")
    st.stop()


with st.form("customer_form"):
    credit_score = st.number_input("Credit score", 300, 900, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 18, 100, 40)
    tenure = st.number_input("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    products = st.number_input("Number of products", 1, 4, 1)
    has_card = st.selectbox("Has credit card?", [0, 1])
    active = st.selectbox("Active member?", [0, 1])

    submitted = st.form_submit_button(
        "Predict estimated salary",
        type="primary",
        width="stretch",
    )

if submitted:
    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active,
    }])

    customer_scaled = preprocessor.transform(customer).astype("float32")
    salary = float(model.predict(customer_scaled, verbose=0)[0][0])

    st.metric("Predicted EstimatedSalary", f"{salary:,.2f}")