import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generate/Load Data
data = {
    'tenure': np.random.randint(1, 72, 1000),
    'MonthlyCharges': np.random.uniform(20, 120, 1000),
    'TotalCharges': np.random.uniform(100, 8000, 1000),
    'Contract_Code': np.random.choice([0, 1, 2], 1000),
    'Churn': np.random.choice([0, 1], 1000)
}
df = pd.DataFrame(data)

# 2. Train Model
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Code']]
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# 3. SAVE THE FILES (This is the missing step!)
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Success! 'model.pkl' and 'scaler.pkl' have been created.")