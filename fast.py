#from datetime import datetime
from fastapi import FastAPI
import pandas as pd
from utils import get_live_status
import pickle
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
#comment
@app.get("/predict")
def prediction():
    print("first")
#load X_input from bucket
    X_pred=get_live_status() #pd.DataFrame live df
#load model. if it is on docker load .load()
    print("cool")
    #model=pickle.load(open("dummy_random_model.pkl","rb"))
    model=joblib.load("dummy_random_model",mmap_mode="r")
    #model = pickle.load(open("dummy_random_model.pkl", 'rb'))
    #model = pickle.load("dummy_random_model.pkl")
    print("second")
# use model.predict(X_input)
    y_pred=model.predict(X_pred["bikes"])
    output=X_pred[["station_number"]]
    output["bikes"]=y_pred
    return output


@app.get("/")
def root():
    return{
    'greeting': 'Hello'
    }
