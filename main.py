from fastapi import FastAPI
from CreditScoreModel import CreditScoreModel
from ModelPredictionClass import PredictionModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
def make_predictions(X: List[CreditScoreModel]):
    print(X)
    df = pd.DataFrame([x.dict() for x in X])
    predicion_model = PredictionModel()
    results = predicion_model.make_predictions(df)
    return results.tolist()