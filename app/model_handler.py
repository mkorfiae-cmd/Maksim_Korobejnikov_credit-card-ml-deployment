from pathlib import Path
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "v1": PROJECT_ROOT / "models" / "model_v1.joblib",
    "v2": PROJECT_ROOT / "models" / "model_v2.joblib",
}
FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

loaded_models = {}


def load_model(model_version="v1"): 
    if model_version not in loaded_models:
        loaded_models[model_version] = joblib.load(MODEL_PATHS[model_version])
    return loaded_models[model_version]

def predict(input_data, model_version="v1"):
    model = load_model(model_version)
    try:
        input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    except KeyError as error:
        raise ValueError(f"Отсутствуют признаки: {error}")
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "model_version": model_version,
    }