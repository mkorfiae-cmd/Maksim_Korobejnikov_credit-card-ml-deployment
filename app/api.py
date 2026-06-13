# Импорты
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from app.model_handler import predict

# Инициализация Flask-приложения
app = Flask(__name__)

# Логирование
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "api_logs.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(message)s"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_request(event_type: str, payload: dict):
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    logger.info(json.dumps(log_record, ensure_ascii=False))
    
# Health-check endpoint
@app.route("/health", methods=["GET"])
def health():
    response = {"status": "healthy", "service": "credit-card-default-prediction",}
    return jsonify(response), 200

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        input_data = request.get_json()
        if input_data is None:
            return jsonify({
                "error": "Request body must be valid JSON."
            }), 400
        model_version = request.args.get("model_version", "v1")
        result = predict(
            input_data=input_data,
            model_version=model_version,
        )

        log_request(
            event_type="prediction_success",
            payload={
                "source": "api",
                "model_version": model_version,
                "prediction": result["prediction"],
                "probability": result["probability"],
            },
        )
        return jsonify(result), 200

    except ValueError as error:
        log_request(
            event_type="prediction_validation_error",
            payload={
                "source": "api",
                "error": str(error),
            },
        )
        return jsonify({
            "error": str(error)
        }), 400

    except FileNotFoundError as error:
        log_request(
            event_type="model_file_error",
            payload={
                "source": "api",
                "error": str(error),
            },
        )
        return jsonify({"error": str(error)}), 500

    except Exception as error:
        log_request(
            event_type="prediction_internal_error",
            payload={
                "source": "api",
                "error": str(error),
            },
        )
        return jsonify({"error": "Internal server error."}), 500

# GUI endpoint
FORM_FIELD_TYPES = {
    "LIMIT_BAL": float,
    "SEX": int,
    "EDUCATION": int,
    "MARRIAGE": int,
    "AGE": int,
    "PAY_0": int,
    "PAY_2": int,
    "PAY_3": int,
    "PAY_4": int,
    "PAY_5": int,
    "PAY_6": int,
    "BILL_AMT1": float,
    "BILL_AMT2": float,
    "BILL_AMT3": float,
    "BILL_AMT4": float,
    "BILL_AMT5": float,
    "BILL_AMT6": float,
    "PAY_AMT1": float,
    "PAY_AMT2": float,
    "PAY_AMT3": float,
    "PAY_AMT4": float,
    "PAY_AMT5": float,
    "PAY_AMT6": float,
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)

@app.route("/gui", methods=["POST"])
def gui_predict():
    try:
        model_version = request.form["model_version"]
        input_data = {
            field: field_type(request.form[field])
            for field, field_type in FORM_FIELD_TYPES.items()
        }
        result = predict(input_data=input_data, model_version=model_version)
        log_request(
            event_type="prediction_success",
            payload={
                "source": "gui",
                "model_version": model_version,
                "prediction": result["prediction"],
                "probability": result["probability"],
            },
        )
        return render_template("index.html", result=result, error=None)

    except Exception as error:
        log_request(
            event_type="gui_prediction_error",
            payload={
                "source": "gui",
                "error": str(error),
            },
        )
        return render_template("index.html", result=None, error=str(error))


# Запуск

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )