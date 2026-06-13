import pytest
from app.api import app


@pytest.fixture
def client():
    # Создаёт тестовый клиент Flask
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    # Корректный JSON-запрос для /predict
    return {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 1,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }


def test_health_returns_status_200(client):
    # /health возвращает статус 200
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "credit-card-default-prediction"


def test_predict_accepts_valid_json(client, valid_payload):
    # /predict принимает корректный JSON
    response = client.post("/predict?model_version=v1", json=valid_payload)
    assert response.status_code == 200


def test_predict_returns_required_fields(client, valid_payload):
    #/predict возвращает нужные поля
    response = client.post(
        "/predict?model_version=v1",
        json=valid_payload,
    )
    data = response.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data

    assert data["model_version"] == "v1"
    assert data["prediction"] in [0, 1]
    assert isinstance(data["probability"], float)


def test_predict_returns_error_for_invalid_json(client):
    # ошибка при неполном JSON
    invalid_payload = {"LIMIT_BAL": 20000, "SEX": 2}
    response = client.post("/predict?model_version=v1", json=invalid_payload)
    data = response.get_json()
    assert response.status_code == 400
    assert "error" in data