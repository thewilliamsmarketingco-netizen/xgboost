from fastapi import Depends, FastAPI, Request, HTTPException
import xgboost as xgb
import pandas as pd
import os
import secrets

app = FastAPI()
API_KEY = os.environ.get("XGBOOST_API_KEY", "")

def require_api_key(req: Request):
    # Keep liveness public, but never serve model requests without a server key.
    if not API_KEY or not API_KEY.strip():
        raise HTTPException(status_code=503, detail="authentication_not_configured")
    authorization = req.headers.get("authorization", "")
    if not secrets.compare_digest(
        authorization.encode("utf-8"), f"Bearer {API_KEY}".encode("utf-8")
    ):
        raise HTTPException(
            status_code=401, detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/train", dependencies=[Depends(require_api_key)])
async def train(req: Request):
    body = await req.json()
    rows = [r for r in body.get("training_data", []) if r.get("was_accurate") is not None]
    if len(rows) < 10:
        return {"status": "not_enough_data", "usable_rows": len(rows)}
    df = pd.DataFrame(rows)
    y = df["was_accurate"].fillna(0).astype(int)
    X = pd.get_dummies(df.drop(columns=["was_accurate"]), dummy_na=True)
    model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, eval_metric="logloss")
    model.fit(X, y)
    model.save_model("/tmp/model.json")
    return {"status": "trained", "rows": len(df), "features": X.shape[1]}

@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(req: Request):
    
    try:
        model = xgb.XGBClassifier()
        model.load_model("/tmp/model.json")
    except Exception as e:
        raise HTTPException(status_code=400, detail="model_not_trained")
    
    body = await req.json()
    data = body.get("data", [])
    if not data:
        raise HTTPException(status_code=400, detail="no_data_provided")
    
    df = pd.DataFrame(data)
    X = pd.get_dummies(df, dummy_na=True)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }
