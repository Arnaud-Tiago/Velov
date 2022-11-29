import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


raw_data_path = '~/.velov/data/raw/'
cleaned_data_path = '~/.velov/data/cleaned/'


def get_station(number:int, start_date:datetime.date='2000-01-01') -> pd.DataFrame:
    """
    input : a station number and a start date
    ---
    return : a dataframe of all timestamps for the given station
    
    """
    
    df = pd.read_csv(raw_data_path+'station'+str(number)+'.csv').drop(columns=['Unnamed: 0']).rename(columns={'0':'time'}).sort_values(by='time').reset_index()
    
    df['time']=pd.to_datetime(df['time'])
    df = df[df['time']>= f"{start_date}"]
    return df

def plot_station(number:int, date_init:datetime.date, date_end:datetime.date):
    """
    input : a station number, a start_date and an end_date
    ---
    plots the number of stands in the given station between the dates
    """
    df  = get_station(number,date_init)
    df = df[(df['time'] > f"{date_init}") & (df['time'] <= f"{date_end}")]
    df = df.set_index('time')
    df = df['stands']
    df.plot()
    
def to_minute(datetime:datetime.date) -> float:
    """
    input : a datetime.timedelta
    ---
    returns : the corresponding timedelta as a float in minutes
    """
    minutes = datetime.days*24*60 + datetime.seconds/60
    return minutes 


def fetch_stats(number:int,start_date:datetime.date='2000-01-01') -> tuple :
    """
    input : a station number and a start_date
    ---
    returns a tuple containing :
        * the nb of observations
        * the lowest timedelta in minutes
        * the highest timedelta in minutes
        * the mean of the timedltas in minutes
        * the median of the timedeltas in minutes
    of the given station from the given start_date
    
    """
    df = get_station(number,start_date)
    nb = df.shape[0]
    timedelta = df['time'].diff().dropna()
    st_min = to_minute(timedelta.min())
    st_mean = to_minute(timedelta.mean())
    st_median = to_minute(timedelta.median())
    st_max = to_minute(timedelta.max())

    return nb, st_min, st_max, st_mean, st_median

def fetch_stats_stations(start_date:datetime.date='2000-01-01') -> pd.DataFrame:
    """
    input : (optionnal) a start_date
    ----
    returns : a dataframe with key info and metrics of stations since the specified start_date
    """
    
    station_info = pd.read_csv(raw_data_path+'stations.csv').drop(columns = ['Unnamed: 0'])
    station_info['nb'] = 0
    station_info['min'] = 0
    station_info['mean'] = 0
    station_info['median'] = 0
    station_info['max'] = 0

    for index, row in station_info.iterrows():
        nb, st_min, st_max, st_mean, st_median = fetch_stats(row['station_number'],start_date)
        station_info.loc[index,'nb'] = nb
        station_info.loc[index,'min'] = st_min
        station_info.loc[index,'mean'] = st_mean
        station_info.loc[index,'median'] = st_median
        station_info.loc[index,'max'] = st_max
        
    return station_info
