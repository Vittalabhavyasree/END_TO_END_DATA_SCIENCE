# =============================
# END-TO-END DATA SCIENCE PROJECT - SINGLE BLOCK
# =============================

# 1️⃣ Install Libraries
!pip install matplotlib seaborn scikit-learn fastapi uvicorn joblib nest-asyncio pyngrok

# 2️⃣ Upload Dataset
from google.colab import files
uploaded = files.upload()  # Upload raw_data.csv

import pandas as pd
df = pd.read_csv("raw_data.csv")
print("\n=== Dataset Head ===")
display(df.head())

# 3️⃣ EDA - Info, Tables
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Summary Statistics ===")
summary_table = df.describe()
display(summary_table)
summary_table.to_csv("summary_statistics.csv")

# 4️⃣ EDA - Graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Target distribution
plt.figure()
sns.countplot(x='income', data=df) # Changed 'target' to 'income'
plt.title("Target Distribution")
plt.savefig("target_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.1)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# 5️⃣ Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('income', axis=1) # Changed 'target' to 'income'
y = df['income'] # Changed 'target' to 'income'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Model Training & Evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# 7️⃣ Save Model & Scaler
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Optional: Download artifacts
from google.colab import files
files.download("model.pkl")
files.download("scaler.pkl")
files.download("target_distribution.png")
files.download("correlation_heatmap.png")
files.download("confusion_matrix.png")
files.download("summary_statistics.csv")

# 8️⃣ FastAPI Deployment in Colab
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Create FastAPI app
app = FastAPI(title="Data Science Prediction API")

class PatientData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array(data.features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled).max()
    return {"prediction": int(prediction), "probability": float(probability)}

# Run API in Colab using ngrok
nest_asyncio.apply()
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

uvicorn.run(app, host="0.0.0.0", port=8000)
