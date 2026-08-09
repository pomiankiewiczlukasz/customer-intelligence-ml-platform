import mlflow
from sklearn.pipeline import Pipeline

MODEL_NAME = "customer-churn-model"
MODEL_VERSION = "1"


def load_model(
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
) -> Pipeline:
    """
    Load a trained ML pipeline from MLflow Model Registry.
    """

    model_uri = f"models:/{model_name}/{model_version}"

    return mlflow.sklearn.load_model(model_uri)