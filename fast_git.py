from fastapi import FastAPI
from utils import get_live_status
import pandas as pd
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import gcsfs
from google.cloud import storage
import joblib

def load_model_from_cloud():
    fs = gcsfs.GCSFileSystem(project="velov-pred")
    client = storage.Client.from_service_account_json("PUT_YOUR_CREDENTIALS")
    bucket = client.bucket("velov_bucket")
    blob = bucket.blob("model/model.pkl")
    with fs.open(f'velov_bucket/{blob.name}') as f:
        model = joblib.load(f)
    return model

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
    X_pred = np.array([np.array(get_live_status()[['bikes','station_number']].set_index('station_number'))]*12)
    X_pred = X_pred.reshape((X_pred.shape[0],X_pred.shape[1]))
    model=load_model_from_cloud()
    y_pred_diff=model.predict(X_pred)
    y_pred_diff = y_pred_diff.reshape((y_pred_diff.shape[1],y_pred_diff.shape[2]))
    y_pred = X_pred + y_pred_diff
    output_dict=pd.DataFrame(y_pred).to_dict()
    return output_dict



@app.get("/")
def root():
    return{
    'greeting': 'Hello'
    }
