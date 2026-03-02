import streamlit as st
import pickle
import pandas as pd
import joblib
import os
model_path = os.path.join("churn_model (1).pkl")
model = joblib.load(model_path)
st.set_page_config(page_title="CUSTOMER CHURN PREDICTOR", layout="wide")
st.title("CUSTOMER CHURN PREDICTION APP")
st.markdown("Predict whether a customer is likely to churn.")
st.subheader("Customer Information")
col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=800.0)
    rough = st.number_input("Rough Charge each month", min_value=0.0, value=70.0)
    contract = st.selectbox("Contract Type",["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
    techsupport = st.selectbox("Tech Support",["Yes", "No"])
    Payment = st.selectbox("PaymentMethod",["Electronic check", "Mailed check", "Bank transfer(automatic)",
                                                   "Credit card(automatic)"])
with col2:    
    multiplelines = st.selectbox("Multiple Lines",["Yes", "No"])
    onlinesecurity = st.selectbox("Online Security",["Yes", "No"])
    backup = st.selectbox("Online Backup",["Yes", "No"])
    protection = st.selectbox("Device Protection",["Yes", "No"])
    tv = st.selectbox("Streaming TV",["Yes", "No"])
    movies = st.selectbox("Streaming Movies",["Yes", "No"])
    billing = st.selectbox("Paperless Billing",["Yes", "No"])
    phone = st.selectbox("PhoneService",["Yes", "No"])
service_count = sum([
1 if phone == "Yes" else 0,
1 if internet == "Yes" else 0,
1 if onlinesecurity == "Yes" else 0,
1 if backup == "Yes" else 0,
1 if protection == "Yes" else 0,
1 if techsupport == "Yes" else 0,
1 if tv == "Yes" else 0,
1 if movies == "Yes" else 0,
1 if multiplelines == "Yes" else 0
])
services = [phone, internet, onlinesecurity, backup,
            protection, techsupport, tv, movies, multiplelines]

service_count = sum([s == "Yes" for s in services])
input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total],
    "Contract": [contract],
    "InternetService": [internet],
    "TechSupport": [techsupport],
    "PaymentMethod":[Payment],
    "MultipleLines":[multiplelines],
    "InternetService":[internet],
    "OnlineSecurity":[onlinesecurity],
    "OnlineBackup":[backup],
    "DeviceProtection":[protection],
    "StreamingTV":[tv],
    "StreamingMovies":[movies],
    "PaperlessBilling":[billing],
    "PhoneService":[phone],
    "avg_monthly_cost":[rough],
    "services_count":[service_count]
    })

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{round(probability*100, 2)}%")
    if probability > 0.7:
        st.error("High Risk Customer")
    elif probability > 0.4:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")
    st.write("Prediction (0 = No churn, 1 = Churn):", prediction)