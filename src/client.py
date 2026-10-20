from model import Model

import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

merchant_categories = [
    "Clothing",
    "Grocery",
    "Food",
    "Electronics",
    "Travel",
]

st.title("Credit Card Fraud Detector - Client")

# amount - transaction hour - device trust score - cardholder age - velocity last 24h

upper_col_1, upper_col_2, upper_col_3, upper_col_4, upper_col_5 = st.columns(5)

with upper_col_1:
    amount = st.number_input(
        "Amount", min_value=1.0, value=100.0, step=100.0, format="%.2f"
    )

with upper_col_2:
    transaction_hour = st.number_input(
        "Transaction hour", min_value=0, max_value=23, value=8, step=1
    )

with upper_col_3:
    device_trust_score = st.number_input(
        "Device trust score", min_value=0, max_value=100, value=50, step=10
    )

with upper_col_4:
    cardholder_age = st.number_input("Cardholder age", min_value=18, value=20, step=1)

with upper_col_5:
    velocity_last_24h = st.number_input(
        "Velocity last 24h", min_value=0, value=0, step=1
    )

# merchant category

merchant_category = st.selectbox("Merchant category", merchant_categories)

# foreign transaction - location mismatch

lower_col_1, lower_col_2 = st.columns(2)

with lower_col_1:
    foreign_transaction = st.checkbox("Foreign transaction")

with lower_col_2:
    location_mismatch = st.checkbox("Location mismatch")

if st.button("Predict", width="stretch"):
    payload = {
        "amount": amount,
        "transaction_hour": transaction_hour,
        "merchant_category": merchant_category,
        "foreign_transaction": foreign_transaction,
        "location_mismatch": location_mismatch,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
    else:
        st.error(response.text)

    if prediction:
        st.error("Fraud transaction detected!")
    else:
        st.success("No fraud detected")
