from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your pre-trained model and scaler
# (Note: You must have saved these using joblib first!)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_code: int

@app.get("/")
def home():
    return {"message": "RetainIQ API is Online"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input to array
    features = np.array([[data.tenure, data.monthly_charges, data.total_charges, data.contract_code]])
    features_scaled = scaler.transform(features)
    
    # Get prediction
    probability = model.predict_proba(features_scaled)[0][1]
    return {"churn_probability": float(probability)}