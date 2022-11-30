import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from datetime import timedelta

root_path = '~/.velov/data/'
raw_data_path = '~/.velov/data/raw/'
cleaned_data_path = '~/.velov/data/cleaned/'

def get_stations_info(source = 'local',save_csv = False) -> pd.DataFrame:
    '''
    This function gathers the main information for all stations, from a local csv file if it exists, from the velov API if not.

    input : a source type (default 'local') and a boolean value indicating whether the results is saved as a csv
    ---
    return : a dataframe containing relevant information for all velov stations (identification number, availability code, number of bike stands, latitude,
    longitude, station name, station pole, station capacity)

    '''
    if source == 'local':
        station_info = pd.read_csv(f'{root_path}stations.csv')
        station_info.drop(columns = ['Unnamed: 0'],inplace = True)

    else:
        url_info = "https://download.data.grandlyon.com/ws/rdata/jcd_jcdecaux.jcdvelov/all.json?compact=false"
        response_numbers = requests.get(url_info)
        station_list = response_numbers.json()['values']
        station_number = []
        station_availability = []
        station_bike_stands = []
        station_address = []
        station_lat = []
        station_lng = []
        station_name = []
        station_pole = []
        station_capacity = []

        for station in station_list:
            station_number.append(station['number'])
            station_availability.append(station['availabilitycode'])
            station_bike_stands.append(station['bike_stands'])
            station_address.append(station['address'])
            station_lat.append(station['lat'])
            station_lng.append(station['lng'])
            station_name.append(station['name'])
            station_pole.append(station['pole'])
            station_capacity.append(station['total_stands']['capacity'])
        station_info = pd.DataFrame([station_number, station_availability, station_bike_stands, station_address,
                station_lat, station_lng,station_name, station_pole, station_capacity],
            index=['station_number','availabilitycode','bike_stands','address','lat','lng','name','pole','capacity']).transpose()
        if save_csv:
            station_info.to_csv(f'{root_path}stations.csv',header=True)


    return station_info


def get_station(number:int, start_date:datetime.date='2000-01-01', source = 'local', save_csv = False) -> pd.DataFrame:
    """
    input : a station number, a start date, a source (default : 'local') and a boolean indicating whether to save as csv or not
    ---
    return : a dataframe of all timestamps for the given station

    """
    if source == 'local':
        # looks for the corresponding local csv file to obtain historical data
        df = pd.read_csv(raw_data_path+'station'+str(number)+'.csv').drop(columns=['Unnamed: 0']).rename(columns={'0':'time'}).sort_values(by='time').reset_index().drop(columns='index')
    elif source == 'api':
        # calls the API to obtain historical data
        url = f'https://download.data.grandlyon.com/ws/timeseries/jcd_jcdecaux.historiquevelov/all.json?field=number&value={number}&compact=true&maxfeatures=1000000'
        response = requests.get(url)
        data = response.json()
        values = data['values']
        horodate =[]
        electrical_bikes =[]
        mechanical_bikes =[]
        stands=[]

        for item in values:
            horodate.append(item['horodate'])
            electrical_bikes.append(item['main_stands']['availabilities']['electricalBikes'])
            mechanical_bikes.append(item['main_stands']['availabilities']['mechanicalBikes'])
            stands.append(item['main_stands']['availabilities']['stands'])
        df = pd.DataFrame(horodate)
        df['electrical_bikes']=electrical_bikes
        df['mechanical_bikes']=mechanical_bikes
        df['stands']=stands
        df = df.rename(columns={0:'time'}).sort_values(by='time').reset_index().drop(columns=['index'])

    df['time']=pd.to_datetime(df['time'],utc=True)
    df = df[df['time']>= f"{start_date}"]
    if save_csv:
        df.to_csv(f'{raw_data_path}/station{number}.csv',header=True)
    return df

def get_cleaned_station(number:int, start_date:datetime.date='2000-01-01', source = 'local', save_csv = False) -> pd.DataFrame:
    """
    input : a station number, a start date, a source (default : 'local') and a boolean indicating whether to save as csv or not
    ---
    return : a dataframe of all timestamps for the given station

    """
    # looks for the corresponding local csv file to obtain historical data
    df = pd.read_csv(cleaned_data_path+'station_'+str(number)+'.csv')


    df['time']=pd.to_datetime(df['time'],utc=True)
    df = df[df['time']>= f"{start_date}"]
    return df


def get_all_data(source = 'local',save_csv = False) -> list:
    '''
    Use this function to obtain all the data at once.
    '''
    station_info = get_stations_info(source = source,save_csv = False)
    return [get_station(number,source,save_csv) for number in station_info.station_number]

def get_last_week_station(number) -> pd.DataFrame:
    '''
    This function returns a DataFrame containing the last seven days of data for the station identified by 'number'. The DataFrame contains:
    index : 5-minutes timestamps
    bikes : total number of available bikes at a given timestamp in the station
    bike_stands : total number of available bike stands at a given timestamp in the station
    '''
    today = (str(datetime.today())[:-7] + 'Z').replace(' ','T')
    previous = (str(datetime.today() - timedelta(days=10))[:-7] + 'Z').replace(' ','T')
    url = f'https://download.data.grandlyon.com/sos/velov?request=GetObservation&service=SOS&version=1.0.0&offering=reseau_velov&procedure=velov-{number}&observedProperty=bikes,bike-stands&eventTime={previous}/{today}&responseFormat=application/json'
    response = requests.get(url)
    df = pd.DataFrame(response.json()['ObservationCollection']['member'][0]['result']['DataArray']['values'],columns=['time','bikes','bike_stands'])
    df.rename(columns = {'bike_stands':'stands'}, inplace=True)
    df['time']=pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.astype(float).astype(int)
    return df

def get_live_status() -> pd.DataFrame:
    '''
    Returns the live status of the whole vélov system as a DataFrame with:
    time : timestamp of the last update for a given station
    station_number : identification number of the station
    bikes : number of available bikes at a given station at the given timestamp
    bike_stands : number of available bike stands at a given station at the given timestamp
    '''
    url_live_status = 'https://transport.data.gouv.fr/gbfs/lyon/station_status.json'
    response = requests.get(url_live_status)
    live_df = pd.DataFrame(response.json()['data']['stations'])
    live_df['time']=live_df['last_reported'].apply(datetime.fromtimestamp)
    live_df = live_df.rename(columns={'num_bikes_available':'bikes','num_docks_available':'bike_stands','station_id':'station_number'})
    live_df = live_df[['time','bikes','bike_stands','station_number']].set_index('time')
    return live_df

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

def find_missing_data(threshold:int, start_date:datetime.date = '2000-01-01') -> pd.DataFrame:
    """
    input : a threshold as an integer (in minutes) and a (optionnal) start date
    ----
    returns : a DataFrame with :
        * the date (truncated at hours level)
        * the timedeltas 
    
    """
    
    station_info = pd.read_csv(raw_data_path+'stations.csv').drop(columns = ['Unnamed: 0'])
    first = True
    
    for index, row in station_info.iterrows():
        df = get_station(row['station_number'], start_date= start_date)
        df['td']=df['time'].diff().map(to_minute)
        df = df[df['td']> threshold]
        df = df[['time','td']].set_index('time')
        df = df.resample('H').median().dropna().reset_index()
        if first == True :
            deltas_df = df.copy()
            first = False
        else :
            deltas_df = pd.concat([deltas_df,df])


    deltas_df['td_bracket_hrs']=deltas_df['td']//60
    deltas_df = deltas_df.groupby(['time','td_bracket_hrs']).count()
    deltas_df = deltas_df.reset_index().set_index('time').sort_values('td_bracket_hrs',ascending = False).rename(columns={'td':'count'})    
    
    
    return deltas_df

def get_elevation_column(dataframe, lat_col_name='lat', lng_col_name='lng'):
    '''
    Takes a DataFrame including latitude and longitude as columns
    Return a Series with elevation (NaN if not found)
    '''

    def get_elevation(lat, lon):
        '''
        Takes lat and lon
        Call an elevation API
        Returns elevation (NaN if not found)

        '''

        url = 'https://api.opentopodata.org/v1/eudem25m?'
        params = {'locations': f'{lat},{lon}'}

        result = requests.get(url=url, params=params, timeout=10).json()
        if 'results' in result.keys():
            return result['results'][0]['elevation']
        return np.nan

    return dataframe.apply(
        lambda row: get_elevation(row[lat_col_name], row[lng_col_name]),
        axis=1)

def get_clean_dataframes() -> pd.DataFrame:
    '''
    returns a dataframe with the number of bikes for each station at any given timestamp from the cleaned historical dataset
    uses the cleaned overall .csv file
    '''
    data = pd.read_csv('~/.velov/data/all_station.csv')
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
    return bikes_df,stands_df
    
