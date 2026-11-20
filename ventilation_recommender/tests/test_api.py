import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data


def test_stats(client):
    response = client.get("/stats")
    assert response.status_code == 200

    data = response.json()
    assert "business_weights" in data
    assert "valid_coverage" in data


def test_business_parameters(client):
    response = client.get("/business-parameters")
    assert response.status_code == 200

    data = response.json()
    assert data["business_weights"]
    assert data["business_characteristics"]


def test_predict_not_found(client):
    response = client.post(
        "/predict",
        json={
            "query": "НЕТ_ТАКОГО_ТОВАРА",
            "k": 5,
            "min_score": 0.0,
            "include_details": False,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query_type"] == "not_found"
    assert data["results"] == []
