from unittest.mock import patch

from fastapi.testclient import TestClient


class FakeModel:
    def predict(self, data):
        return [0]

    def predict_proba(self, data):
        return [[0.75, 0.25]]


with patch("src.models.load_model.load_model", return_value=FakeModel()):
    from src.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 15,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 75.10,
        "TotalCharges": 1151.55,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    data = response.json()

    assert data["churn_prediction"] == 0
    assert data["churn_probability"] == 0.25