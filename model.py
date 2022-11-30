import pandas as pd
from datetime import timedelta
from sklearn.metrics import  accuracy_score,mean_squared_error,recall_score,f1_score,precision_score
from velov import utils

station_info = utils.get_stations_info()


def classify(bikes : int,bike_stands : int, capacity : int,tolerance_level = 0.1) -> int:
    '''
    input : number of available bikes and bike stands, station capacity and tolerance level to deem a station nearly empty or nearly full
    ---
    result : number indicating station status:
        -  0 - empty - there is no bike available;
        -  1 - nearly empty - there is less than a percentage based on the tolerance level of the bikes available;
        -  2 - normal - there is both a satisfying number of bikes and bike stands at the station;
        -  3 - nearly full - there is less than a percentage based on the tolerance level of the bike stands available;
        -  4 - full - the station is full, there is no bike stand available.
    '''
    if bikes <= 0:
        return 0
    if bikes < tolerance_level * capacity:
        return 1
    if bike_stands <=0:
        return 4
    if bike_stands < tolerance_level * capacity:
        return 3
    return 2

def classify_station(station_data : pd.DataFrame) -> pd.DataFrame:
    classified_station_data = station_data.copy()
    classified_station_data['status_code']= classified_station_data.apply(lambda x: classify(x.bikes,x.bike_stands,utils.get_stations_info().capacity.values[0]),axis = 1)
    classified_station_data['status'] = classified_station_data['status_code'].map({0:"empty",1:"nearly empty",2:"OK",3:"nearly full",4:'full'})
    classified_station_data['is_empty'] = classified_station_data['status_code'].map({0:1,1:0,2:0,3:0,4:0})
    classified_station_data['is_full'] = classified_station_data['status_code'].map({0:0,1:0,2:0,3:0,4:1})
    
    return classified_station_data


def predict(station_data : pd.DataFrame,n_min = 15,n_days = 7) -> pd.DataFrame:
    '''
    input : station_data (classified), number of minutes (multiple of 5) of offset, with the number of days of data considered.
    ---
    return : baseline model prediction
    '''
    y = station_data.copy().reset_index().iloc[-n_days*24*12:]
    y_pred = station_data.copy().reset_index().iloc[int(-(n_days*24*12+n_min/5)):-int(n_min/5)]
    y_pred['time']=y_pred['time'] + timedelta(minutes = n_min)
    y_pred = y_pred.reset_index().drop(columns = 'index')
    return y,y_pred

def compute_metrics(classified_station_data,n_days = 7):
    mse = []
    precision_empty = []
    precision_full = []  
    accuracy = []
    step =[]
    
    steps = range(1,13)
    
    for i in steps:
        y,y_pred = predict(classified_station_data,n_min = i*5,n_days=n_days)
        mse.append(mean_squared_error(y.bikes,y_pred.bikes))
        precision_empty.append(precision_score(y.is_empty,y_pred.is_empty))
        precision_full.append(precision_score(y.is_full,y_pred.is_full))
        accuracy.append(accuracy_score(y.status_code,y_pred.status_code))
        step.append(f'{str(int((i*5)))} m')    
    print(mse)
  
    result=pd.DataFrame(index=['step','mse','precision_empty','precision_full','accuracy'],data=[step,mse,precision_empty,precision_full,accuracy]).transpose()
       
    return result
    