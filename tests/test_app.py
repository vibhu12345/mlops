"""Module providing test cases"""

import pytest
from app import app

@pytest.fixture(name="client")
def client_fixture():
    """Function exposing test client"""
    with app.test_client() as testclient:
        yield testclient

def test_predict(client):
    """Function testing predict method"""
    response = client.post("/predict", json={"features": [
        63,
        1,
        3,
        145,
        233,
        1,
        0,
        150,
        0,
        2.1,
        3,
        0.0,
        3.0
    ]})
    assert response.status_code == 200
    assert "prediction" in response.get_json()

def test_invalid_input(client):
    """Function testing invalid input"""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()
