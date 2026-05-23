from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json

app = FastAPI(title="Armenian Genocide Survival Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", os.getenv("FRONTEND_URL", "")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_here = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(_here, "best_model.pkl"))
model = joblib.load(MODEL_PATH)

# Pledge counter — in-memory, seeded from file if present
_PLEDGE_FILE = os.path.join(_here, "pledges.json")
_BASE = 1847

def _load_count() -> int:
    try:
        return json.loads(open(_PLEDGE_FILE).read())["count"]
    except Exception:
        return _BASE

def _save_count(n: int):
    try:
        open(_PLEDGE_FILE, "w").write(json.dumps({"count": n}))
    except Exception:
        pass

_pledge_count = _load_count()


class PredictRequest(BaseModel):
    age: int
    gender: str
    religion: str
    ethnicity: str
    location: str
    occupation: str
    socioeconomic_status: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.model_dump()])
    pred = model.predict(df)[0]
    probability = max(1, min(99, int(round(float(pred)))))
    return {"survival_probability": probability}


@app.get("/pledge")
def get_pledge():
    return {"count": _pledge_count}


@app.post("/pledge")
def post_pledge():
    global _pledge_count
    _pledge_count += 1
    _save_count(_pledge_count)
    return {"count": _pledge_count}
