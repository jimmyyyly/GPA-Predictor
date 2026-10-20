from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_gpa_in_range():
    res = client.post("/predict", json={"math_sat": 650, "verbal_sat": 600})
    assert res.status_code == 200
    assert 0.0 <= res.json()["predicted_gpa"] <= 4.0


def test_higher_scores_predict_higher_gpa():
    low = client.post("/predict", json={"math_sat": 520, "verbal_sat": 500}).json()
    high = client.post("/predict", json={"math_sat": 710, "verbal_sat": 700}).json()
    assert high["predicted_gpa"] > low["predicted_gpa"]


def test_rejects_invalid_scores():
    res = client.post("/predict", json={"math_sat": 900, "verbal_sat": 600})
    assert res.status_code == 422
