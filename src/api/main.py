import pandas as pd
from fastapi import FastAPI

from src.api.schemas import CustomerData
from src.models.load_model import load_model

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a model from MLflow.",
    version="1.0.0",
)

model = load_model()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    customer_data = customer.model_dump()

    input_data = pd.DataFrame([customer_data])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
    }