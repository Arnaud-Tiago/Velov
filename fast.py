#from datetime import datetime
from fastapi import FastAPI
import pandas as pd
from utils import get_stations_info, get_live_status
from model import predict
from fastapi.middleware.cors import CORSMiddleware
#SERVE FUNZIONE DI PRED

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def prediction():
    X_pred=get_live_status() #pd.DataFrame live df

    y_pred=predict(X_pred,n_min = 15,n_days = 7)
    return y_pred


@app.get("/")
def root():
    return{
    'greeting': 'Hello'
    }
