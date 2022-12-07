from fastapi import FastAPI
from utils import get_live_status
import pickle
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

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
    X_pred=get_live_status()
    model=pickle.load(open("dummy_random_model_3.pkl",'rb'))
    y_pred=model.predict(X_pred["bikes"])
    output=X_pred[["station_number"]]
    output["bikes"]=y_pred
    output=output.set_index(["station_number"])
    output_dict = output.to_dict()
    return output_dict


@app.get("/")
def root():
    return{
    'greeting': 'Hello'
    }
