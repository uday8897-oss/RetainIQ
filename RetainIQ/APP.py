import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="RetainIQ | ChurnGuard AI", layout="wide")

# --- DATA ENGINE (Synthetic Data for Instant Run) ---
@st.cache_data
def get_data():
    np.random.seed(42)
    data = {
        'Tenure': np.random.randint(1, 72, 1000),
        'MonthlyCharges': np.random.uniform(20, 120, 1000),
        'TotalCharges': np.random.uniform(100, 8000, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
        'Churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    
    # Simple Preprocessing
    le = LabelEncoder()
    df['Contract_Encoded'] = le.fit_transform(df['Contract'])
    return df, le

df, le = get_data()

# --- MODEL TRAINING ---
# We train the model every time the app starts (or use @st.cache_resource)
X = df[['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Encoded']]
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Customer Profile")
st.sidebar.markdown("Enter details to predict churn risk.")

u_tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 24)
u_monthly = st.sidebar.number_input("Monthly Charges ($)", 20.0, 150.0, 65.0)
u_total = st.sidebar.number_input("Total Charges ($)", 20.0, 10000.0, 1500.0)
u_contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

# Map contract back to encoding
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
u_contract_encoded = contract_map[u_contract]

# --- MAIN INTERFACE ---
st.title("🚀 RetainIQ: Predictive Retention Dashboard")
st.markdown("This dashboard uses the **ChurnGuard AI** classification model to identify high-risk customers.")

col1, col2 = st.columns([1, 1])

# --- PREDICTION LOGIC ---
user_input = np.array([[u_tenure, u_monthly, u_total, u_contract_encoded]])
user_input_scaled = scaler.transform(user_input)
prediction_proba = model.predict_proba(user_input_scaled)[0][1]
risk_percent = round(prediction_proba * 100, 2)

with col1:
    st.subheader("🎯 Risk Analysis")
    if risk_percent > 70:
        st.error(f"CRITICAL CHURN RISK: {risk_percent}%")
    elif risk_percent > 40:
        st.warning(f"MODERATE CHURN RISK: {risk_percent}%")
    else:
        st.success(f"STABLE CUSTOMER: {risk_percent}% Risk")
    
    st.progress(risk_percent / 100)
    
    # Visualizing Probability
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.barplot(x=["Stay", "Churn"], y=model.predict_proba(user_input_scaled)[0], palette="viridis")
    plt.ylabel("Probability")
    st.pyplot(fig)

with col2:
    st.subheader("📊 Feature Importance")
    st.write("What factors are driving this prediction?")
    feat_importances = pd.Series(model.feature_importances_, index=['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract'])
    st.bar_chart(feat_importances)

# --- STRATEGY SIMULATOR ---
st.divider()
st.subheader("💡 RetainIQ 'What-If' Strategy Simulator")
st.info("Check how changing the contract affects this customer's risk score.")

# Calculate "What-if" for a 2-Year Contract
sim_input = np.array([[u_tenure, u_monthly, u_total, 2]]) # 2 is 'Two year'
sim_input_scaled = scaler.transform(sim_input)
sim_proba = model.predict_proba(sim_input_scaled)[0][1]
sim_risk = round(sim_proba * 100, 2)

s_col1, s_col2 = st.columns(2)
s_col1.metric("Current Risk", f"{risk_percent}%")
s_col2.metric("Risk with 2-Year Contract", f"{sim_risk}%", f"-{round(risk_percent - sim_risk, 2)}%")

st.write("---")
st.caption("Developed by Alli | RetainIQ v1.0")
import requests

# When the user clicks the button:
payload = {
    "tenure": u_tenure,
    "monthly_charges": u_monthly,
    "total_charges": u_total,
    "contract_code": u_contract_encoded
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
result = response.json()
st.write(f"Risk from API: {result['churn_probability'] * 100}%")