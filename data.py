#save data from cloud
import pandas as pd
from google.cloud import storage
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from datetime import timedelta

from utils import get_stations_info, get_last_week_station
from params import LOCAL_DATA_PATH_CLEAN, LOCAL_DATA_PATH_RAW, LOCAL_ROOT_PATH, BUCKET_NAME

root_path = LOCAL_ROOT_PATH
raw_data_path = LOCAL_DATA_PATH_RAW
cleaned_data_path = LOCAL_DATA_PATH_CLEAN


def save_data(df:pd.DataFrame, table_name:str, clean:bool=False ,destination='local'):
    """
    Saves a DataFrame df
        * name it with the table_name argument
        * store it in raw folder if clean is False, else store it in clean Folder
        * destination may be 'local' or 'cloud'
    
    """
    
    valid = {'local','cloud'}
    if destination not in valid:
        raise ValueError("Error : destination must be one of %r." % valid)
    
    if destination=="cloud":
        client = storage.Client() # use service account credentials
        export_bucket = client.get_bucket(BUCKET_NAME) #define bucket
        if clean:
            blob_name = 'cleaned/'+table_name
        else :
            blob_name = 'raw/'+table_name        
        # HARD STOP
        export_bucket.blob(blob_name).upload_from_string(df.to_csv(),"text/csv")
        print(f"DataFrame successfully saved on the cloud at {BUCKET_NAME}/{blob_name}")


    else:
        if clean :
            path = cleaned_data_path+table_name+'.csv'
        else :
            path = raw_data_path+table_name+'.csv'
        df.to_csv(path)
        print(f"DataFrame successfully saved locally on {path}")  
    
    return None


def load_data(table_name:str, clean:bool=False, provenance='local') -> pd.DataFrame:
    """
    input :
        * table_name as a str
        * clean as a boolean : determines if the table is to be load from cleaned or raw folder
        * provenance : can be 'local' or 'cloud' 
    ------
    returns : a pd.DataFrance
    """    
    valid = {'local','cloud'}
    if provenance not in valid:
        raise ValueError("Error : destination must be one of %r." % valid)
    
    if provenance == 'local':
        if clean : 
            path = cleaned_data_path+table_name+'.csv'
        else :
            path = raw_data_path+table_name+'.csv'
        df = pd.read_csv(path)
        print(f"DataFrame successfully loaded from {path}.")

    else :
        client = storage.Client() # use service account credentials
        bucket = client.bucket(BUCKET_NAME)
        if clean : 
            blob_name = 'cleaned/'+table_name
        else :
            blob_name = 'raw/'+table_name   
        df = pd.read_csv('gs://'+BUCKET_NAME+'/'+blob_name)     
        print(f"DataFrame successfully loaded from {BUCKET_NAME}/{blob_name}.")
    
    return df


def increment_data(source='local', save=False, verbose=1):
    """
    takes the otiginal all_stations_live df and increment it with all available unseen data
    verbose = 0 will only display loading and saving messages,
    verbose = 1 only gives key messages,
    verbose = 2 displays the error (missing stations)
    
    """
    valid = {'local','cloud'}
    if source not in valid:
        raise ValueError("Error : source must be one of %r." % valid)
    
    station_df = get_stations_info(source = source)
    nb_updated=0
    nb_errors = 0
    
    clean_df = load_data(table_name= 'all_stations_live', clean=True, provenance=source)
    
    if verbose >0 :
        print(f"Nombre de lignes dans le DataFrame initial : {clean_df.shape[0]:_}.".replace('_',' '))
        print("Work in progress ... This may take a few minutes ...")

    first = True

    for nb in station_df['station_number']:
        last_ts = clean_df.query(f"station_number == {nb}")['time'].max()
        try :
            st_df = get_last_week_station(nb).reset_index()
            st_df['station_number']=nb
            st_df = st_df[st_df['time'] > last_ts]
            if first :
                new_df = st_df.copy()
                first = False
            else :
                new_df = pd.concat([new_df, st_df]) 
            nb_updated +=1   
        except :
            nb_errors += 1
            if verbose > 1:
                print(f"Error on station n° {nb} - not available")

    if verbose > 0:
        print(f"Nombre de stations updatées : {nb_updated}. \nNombre de stations non disponibles : {nb_errors}.")
        print(f"Nombre de lignes dajoutées : {new_df.shape[0]:_}.".replace('_',' '))
    
    clean_df = pd.concat([clean_df,new_df])
    clean_df['time']=pd.to_datetime(clean_df['time'],utc = True)  
    clean_df.set_index('time',inplace=True)

    if verbose > 0:
        print(f"Nombre de lignes dans le DataFrame final : {clean_df.shape[0]:_}.".replace('_',' '))
    
    if save:
        save_data(clean_df, table_name= 'all_stations_live', clean=True ,destination=source)
    
    return clean_df
    