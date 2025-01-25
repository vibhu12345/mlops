import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    response = client.post("/predict", json={"features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]})
    assert response.status_code == 200
    assert "prediction" in response.get_json()

def test_invalid_input(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()
