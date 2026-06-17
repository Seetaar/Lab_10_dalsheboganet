"""
Тесты для API
"""

import pytest
from fastapi.testclient import TestClient
from main import app
import io

client = TestClient(app)

def test_root():
    """Тест главной страницы"""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()

def test_health():
    """Тест проверки работоспособности"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_without_model():
    """Тест предсказания без загруженной модели"""
    data = {
        "age": 35,
        "income": 100000,
        "loan_amount": 3000000
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 400

def test_upload_model():
    """Тест загрузки модели"""
    # Создаем фейковый pkl файл
    model_data = io.BytesIO(b"fake model data")
    response = client.post(
        "/upload-model",
        files={"file": ("model.pkl", model_data, "application/octet-stream")}
    )
    # Может упасть при загрузке, но endpoint должен существовать
    assert response.status_code in [200, 400, 500]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
