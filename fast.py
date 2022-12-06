#from datetime import datetime
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
#comment
@app.get("/predict")
def prediction():
    #load X_input from bucket
    X_pred=get_live_status() #pd.DataFrame live df
    #load model. if it is on docker load .load()
    #model=pickle.load(open("dummy_random_model.pkl","rb"))
    #pickle_file = 'dummy_random_model_test.joblib'
    #with open(pickle_file, 'rb') as f:  model=joblib.load(f)
    model=pickle.load(open("dummy_random_model_3.pkl",'rb'))
    #model=joblib.load("dummy_random_model_2.pkl")
    #model=tf.keras.models.load_model("dummy_random_model.keras")
    #model = pickle.load(open("dummy_random_model.pkl", 'rb'))
    #model = pickle.load("dummy_random_model.pkl")
    # use model.predict(X_input)
    y_pred=model.predict(X_pred["bikes"])
    output=X_pred[["station_number"]]
    output["bikes"]=y_pred
    output=output.set_index(["station_number"])
    #one_station_pred = output['bikes'][0]
    #one_station_pred = float(output[output['station_number'] == 2001]['bikes'])
    #return {'pred': 'nothing'}
    output_dict = output.to_dict()
    return output_dict
#{'pred': float(one_station_pred)}


@app.get("/")
def root():
    return{
    'greeting': 'Hello'
    }
