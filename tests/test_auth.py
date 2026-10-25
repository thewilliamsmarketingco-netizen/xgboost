import importlib

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def service(monkeypatch):
    # Reload so unset/empty tests exercise the actual environment initialization.
    def configure(key):
        if key is None:
            monkeypatch.delenv("XGBOOST_API_KEY", raising=False)
        else:
            monkeypatch.setenv("XGBOOST_API_KEY", key)
        module = importlib.reload(main)
        return module, TestClient(module.app)

    yield configure
    # Avoid leaving a test credential cached in the imported application.
    monkeypatch.undo()
    importlib.reload(main)


@pytest.mark.parametrize("endpoint", ["/train", "/predict"])
@pytest.mark.parametrize("key", [None, "", " \t "])
@pytest.mark.parametrize("authorization", [None, "Bearer ", "Bearer test-key"])
def test_unconfigured_auth_fails_closed(service, monkeypatch, endpoint, key, authorization):
    module, client = service(key)
    model_calls = []
    monkeypatch.setattr(module.xgb, "XGBClassifier", lambda **kwargs: model_calls.append(kwargs))
    headers = {} if authorization is None else {"Authorization": authorization}

    # Invalid JSON proves authentication happens before parsing or model access.
    response = client.post(endpoint, content="invalid json", headers=headers)

    assert response.status_code == 503
    assert response.json() == {"detail": "authentication_not_configured"}
    assert model_calls == []


@pytest.mark.parametrize("endpoint", ["/train", "/predict"])
@pytest.mark.parametrize("authorization", [
    None, "", "Bearer ", "Bearer wrong-key", "Basic test-key", "test-key",
])
def test_missing_or_invalid_credentials_rejected(service, monkeypatch, endpoint, authorization):
    module, client = service("test-key")
    model_calls = []
    monkeypatch.setattr(module.xgb, "XGBClassifier", lambda **kwargs: model_calls.append(kwargs))
    headers = {} if authorization is None else {"Authorization": authorization}

    response = client.post(endpoint, content="invalid json", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert model_calls == []


@pytest.mark.parametrize("key", [None, "", " \t ", "test-key"])
def test_health_remains_public(service, key):
    _, client = service(key)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_credentials_preserve_training_and_prediction(service, monkeypatch, tmp_path):
    module, client = service("test-key")
    classifier = module.xgb.XGBClassifier
    save_model, load_model = classifier.save_model, classifier.load_model
    model_path = tmp_path / "model.json"
    monkeypatch.setattr(classifier, "save_model", lambda self, path: save_model(self, model_path))
    monkeypatch.setattr(classifier, "load_model", lambda self, path: load_model(self, model_path))
    headers = {"Authorization": "Bearer test-key"}

    insufficient = client.post("/train", json={"training_data": []}, headers=headers)
    assert insufficient.status_code == 200
    assert insufficient.json() == {"status": "not_enough_data", "usable_rows": 0}

    untrained = client.post("/predict", json={"data": [{"feature": 1}]}, headers=headers)
    assert untrained.status_code == 400
    assert untrained.json() == {"detail": "model_not_trained"}

    trained = client.post("/train", json={"training_data": [
        {"feature": i, "was_accurate": i % 2} for i in range(12)
    ]}, headers=headers)
    assert trained.status_code == 200
    assert trained.json() == {"status": "trained", "rows": 12, "features": 1}
    assert model_path.is_file()

    predicted = client.post("/predict", json={"data": [{"feature": 1}]}, headers=headers)
    assert predicted.status_code == 200
    assert len(predicted.json()["predictions"]) == 1
    assert len(predicted.json()["probabilities"][0]) == 2
