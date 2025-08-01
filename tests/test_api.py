import pytest
import requests
from fastapi.testclient import TestClient
from app.main import app

mock_request_body = {
    "historical_sales": [50.0, 55.0, 60.0, 65.0, 70.0],
    "forecast_steps": 5
}

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Retail Demand Forecasting API! Use /predict/{category_name} for forecasts."}

def test_predict_no_api_key(client):
    response = client.post("/predict/Furniture", json=mock_request_body)
    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}

def test_predict_incorrect_api_key(client):
    incorrect_headers = {"X-API-Key": "incorrect_password"}
    response = client.post("/predict/Furniture", json=mock_request_body, headers=incorrect_headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized: Invalid API Key"}

def test_predict_correct_api_key(client):
    correct_headers = {"X-API-Key": "password"}
    response = client.post("/predict/Furniture", json=mock_request_body, headers=correct_headers)
    assert response.status_code == 200
    assert "forecast" in response.json()
    assert isinstance(response.json()["forecast"], list)

def test_incorrect_category(client):
    correct_headers = {"X-API-Key": "password"}
    response = client.post("/predict/weather", json=mock_request_body, headers=correct_headers)
    assert response.status_code == 404
    assert response.json() == {"detail": "Model for category 'weather' not found or not loaded."}

def test_invalid_data(client):
    correct_headers = {"X-API-Key": "password"}
    request_body = {"junk": [1, 2, 3, 4]}
    response = client.post("/predict/Furniture", json=request_body, headers=correct_headers)
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "missing",
                "loc": [
                    "body",
                    "historical_sales"
                ],
                "msg": "Field required",
                "input": {
                    "junk": [1, 2, 3, 4]
                }
            }
        ]
    }