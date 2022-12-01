import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from datetime import timedelta

from utils import get_stations_info, get_last_week_station
from data import load_data, save_data
from params import LOCAL_DATA_PATH_CLEAN, LOCAL_DATA_PATH_RAW, LOCAL_ROOT_PATH

root_path = LOCAL_ROOT_PATH
raw_data_path = LOCAL_DATA_PATH_RAW
cleaned_data_path = LOCAL_DATA_PATH_CLEAN

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
    
