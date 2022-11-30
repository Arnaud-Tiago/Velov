import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from datetime import timedelta

from utils import get_stations_info, get_last_week_station

root_path = '~/.velov/data/'
raw_data_path = '~/.velov/data/raw/'
cleaned_data_path = '~/.velov/data/cleaned/'

def increment_data(source='local', save=False, verbose=False):
    """
    takes the otiginal cleaned_df and increment it with all available unseen data
    """
    
    station_df = get_stations_info(source = source)
    nb_updated=0
    nb_errors = 0
    
    if source =='local':
        clean_df = pd.read_csv(cleaned_data_path+'all_station.csv')
    else :
        pass # YOUR CODE HERE
    
    if verbose :
        print(f"Nombre de lignes dans le DataFrame initial : {clean_df.shape[0]:_}.".replace('_',' '))

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
            if verbose :
                print(f"Error on station n° {nb} - not available")

    if verbose :
        print(f"Nombre de stations updatées : {nb_updated}. \nNombre de stations non disponibles : {nb_errors}.")
        print(f"Nombre de lignes dajoutées : {new_df.shape[0]:_}.".replace('_',' '))
    
    clean_df = pd.concat([clean_df,new_df])
    clean_df['time']=pd.to_datetime(clean_df['time'],utc = True)  

    if verbose :
        print(f"Nombre de lignes dans le DataFrame final : {clean_df.shape[0]:_}.".replace('_',' '))
    
    if save:
        pass # YOUR CODE HERE
    
    return clean_df
    
