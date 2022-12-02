import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from utils import get_station, get_stations_info


root_path = '~/.velov/data/'
raw_data_path = '~/.velov/data/raw/'
cleaned_data_path = '~/.velov/data/cleaned/'

def clean_station(number:int, start_date:datetime.date='2000-01-01', step:int=5, last_ts:datetime.date=None) -> pd.DataFrame:
    """
    input : a station number, a start date, and a step in minutes
    ---
    returns : a data frame with data every 5 minutes 
    """
    step_str = f'{step}min'    
    df = get_station(number, start_date)
    if 'mechanical_bikes' in df.columns and 'electrical_bikes' in df.columns:
        df['bikes']=df['mechanical_bikes'] + df['electrical_bikes']
        df.drop(columns = ['electrical_bikes','mechanical_bikes'], inplace=True)
    return df.resample(step_str, on='time').mean().fillna(method='ffill')


def compile_cleaned_df(source='local',save_csv=False) -> pd.DataFrame:
    """
    return the concatenated cleaned flat dataframe
    """
    
    if source == 'local':
        station_df = get_stations_info()
        first = True
        for number in station_df['station_number']:
            st_df = get_cleaned_station(number)
            st_df['station_number']=number
            if first :
                cl_df = st_df.copy()
                first = False
            else : cl_df = pd.concat([cl_df,st_df])
        if save_csv:
            cl_df.to_csv(cleaned_data_path+'all_station.csv', index = 'time')
    else :
        pass # YOUR CODE HEFE
    
    return cl_df
    
    

def get_cleaned_station(number:int, start_date:datetime.date='2000-01-01', source = 'local') -> pd.DataFrame:
    """
    input : a station number, a start date, a source (default : 'local') and a boolean indicating whether to save as csv or not
    ---
    return : a dataframe of all timestamps for the given station

    """
    if source == 'local':
        # looks for the corresponding local csv file to obtain historical data
        df = pd.read_csv(cleaned_data_path+'station_'+str(number)+'.csv')
    elif source == 'big_query':
        # YOUR CODE HERE
        pass

    df = df[df['time']>= f"{start_date}"]
    df.set_index('time',inplace=True)

    return df

def fetch_stats_cleaned_station(number:int,start_date:datetime.date='2000-01-01') -> tuple :
    """
    input : a station number and a start_date
    ---
    returns a tuple containing :
        * the oldest datapoint
        * the newest datapoint
        * the number of datapoints
    of the given station from the given start_date

    """
    df = get_cleaned_station(number,start_date)
    nb = df.shape[0]
    oldest = list(df.index)[0]
    newest = list(df.index)[-1]

    return oldest, newest, nb 

def fetch_stats_cleaned_stations(start_date:datetime.date='2000-01-01') -> pd.DataFrame:
    """
    input : (optionnal) a start_date
    ----
    returns : a dataframe with key info and metrics of stations since the specified start_date
    """

    station_info = pd.read_csv(raw_data_path+'stations.csv').drop(columns = ['Unnamed: 0'])
    station_info['oldest'] = 0
    station_info['newest'] = 0
    station_info['nb'] = 0

    for index, row in station_info.iterrows():
        oldest, newest, nb = fetch_stats_cleaned_station(row['station_number'],start_date)
        station_info['oldest'] = oldest
        station_info['newest'] = newest
        station_info['nb'] = nb

    return station_info

def plot_clean_vs_dirty(station_number:int, day:datetime.date):

    s_day_str = f"{day.year}-{day.month}-{day.day}"
    e_day_str = f"{day.year}-{day.month}-{day.day+1}"
    
    df_dirty  = get_station(station_number)
    df_dirty = df_dirty[(df_dirty['time'] > s_day_str) & (df_dirty['time'] <= e_day_str)]
    df_dirty = df_dirty[['time','stands']].set_index('time')

    df_clean = get_cleaned_station(station_number)
    df_clean.reset_index(inplace=True)
    df_clean['time']=pd.to_datetime(df_clean['time'],utc=True)
    df_clean = df_clean[(df_clean['time'] > s_day_str) & (df_clean['time'] <= e_day_str)]
    df_clean = df_clean[['time','stands']].set_index('time')

    plt.figure(figsize=(24,8));
    plt.title(f"Station number : {station_number}, on {s_day_str}")
    plt.plot(df_dirty,color='r');
    plt.plot(df_clean);

"""

Fonction non utilisée

def get_clean_dataframes() -> pd.DataFrame:
    '''
    returns two dataframes:
        * one with bikes
        * one with available stands
    '''
    data = pd.read_csv(cleaned_data_path+'all_stations_hist.csv')
    first_station = data.station_number.unique()[0]
    df = data[data['station_number']==first_station]
    bikes_df = df[['time','bikes']].rename(columns = {'bikes':f'station_{first_station}'})
    stands_df = df[['time','stands']].rename(columns = {'stands':f'station_{first_station}'})
    for number in np.delete(data.station_number.unique(),0):
        temp_df = data[data['station_number']==number]
        temp_bikes_df = temp_df[['time','bikes']].rename(columns={'bikes':f'station_{number}'})
        temp_stands_df = temp_df[['time','stands']].rename(columns ={'stands':f'station_{number}'})
        bikes_df = bikes_df.merge(temp_bikes_df,on='time',how='outer')
        stands_df = stands_df.merge(temp_stands_df,on='time',how='outer')
    return bikes_df.fillna(method='ffill').fillna(method = 'backfill'),stands_df.fillna(method='ffill').fillna(method = 'backfill')
"""

def get_clean_bikes_dataframe() -> pd.DataFrame:
    '''
    returns the cleaned_bike_dataframe
    '''
    data = pd.read_csv(cleaned_data_path+'all_stations_hist.csv')
    first_station = data.station_number.unique()[0]
    df = data[data['station_number']==first_station]
    bikes_df = df[['time','bikes']].rename(columns = {'bikes':f'station_{first_station}'})
    for number in np.delete(data.station_number.unique(),0):
        temp_df = data[data['station_number']==number]
        temp_bikes_df = temp_df[['time','bikes']].rename(columns={'bikes':f'station_{number}'})
        bikes_df = bikes_df.merge(temp_bikes_df,on='time',how='outer')
    return bikes_df.fillna(method='ffill').fillna(method = 'backfill')

def get_clean_stands_dataframe() -> pd.DataFrame:
    '''
    returns the cleaned_stands_dataframe
    '''
    data = pd.read_csv(cleaned_data_path+'all_stations_hist.csv')
    first_station = data.station_number.unique()[0]
    df = data[data['station_number']==first_station]
    stands_df = df[['time','stands']].rename(columns = {'stands':f'station_{first_station}'})
    for number in np.delete(data.station_number.unique(),0):
        temp_df = data[data['station_number']==number]
        temp_stands_df = temp_df[['time','stands']].rename(columns={'stands':f'station_{number}'})
        stands_df = stands_df.merge(temp_stands_df,on='time',how='outer')
    return stands_df.fillna(method='ffill').fillna(method = 'backfill')
