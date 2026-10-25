# XGBoost model service

Set `XGBOOST_API_KEY` to a nonempty, non-whitespace secret in the deployment's
environment before starting the service. Store it in the hosting provider's secret
manager; never commit it or print its value. Restart workers after changing it,
since the key is loaded when `main` is imported.

`POST /train` and `POST /predict` require `Authorization: Bearer <key>`.
Missing or incorrect credentials receive `401` with a Bearer challenge. If the
server key is absent, empty, or whitespace-only, both endpoints return `503`
before reading request data or accessing the model. `GET /` stays public and
returns `{"status":"ok"}` as a liveness check; it does not indicate authentication
readiness.

The Procfile starts Uvicorn on the host-provided `PORT`. Before rollout, verify
in the hosting provider that the service has a nonblank `XGBOOST_API_KEY`, deploy
the correction, restart workers, and check public liveness plus rejected and
authenticated model requests using synthetic data. Do not log authorization
headers. This repository does not identify a production host or establish its
current environment configuration.

Run tests in a virtual environment with Python 3.11 or 3.12:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The development requirements pin NumPy 1.x for compatibility with pandas 2.1.3
and HTTPX 0.27 for the FastAPI/Starlette TestClient. Tests use synthetic data and
an isolated temporary model file.
