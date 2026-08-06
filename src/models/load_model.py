from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/churn_model.pkl")


def load_model(
    path: Path = MODEL_PATH,
) -> Pipeline:
    """
    Load trained ML pipeline.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)