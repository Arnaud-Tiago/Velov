import pandas as pd
import pydeck as pdk

def create_scatter_layer(data, color):
    '''
    Takes a DataFrame with 'lng' and 'lat' columns
    Create a scatter layer
    '''
    return pdk.Layer('ScatterplotLayer',
                     data=data,
                     get_position=['lng', 'lat'],
                     get_radius=50,
                     get_fill_color=color)


def classify(bikes: int,
             bike_stands: int,
             capacity: int,
             tolerance_level=0.1) -> int:
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
    if bike_stands <= 0:
        return 4
    if bike_stands < tolerance_level * capacity:
        return 3
    return 2


def get_sub_dataframe_from_status(data, status):
    '''
    Select all rows with a given status
    Hint :
    - Must have a columns names 'status'
    - Status values should be in 'empty', 'nearly empty', 'OK', 'nearly full', 'full'
    Returns a DataFrame
    '''
    return data[data['status'] == status]


def classify_station(station_data: pd.DataFrame) -> pd.DataFrame:
    classified_station_data = station_data.copy()
    classified_station_data['status_code'] = classified_station_data.apply(
        lambda x: classify(x.bikes, x.bike_stands,
                           station_data.capacity.values[0]),
        axis=1)
    classified_station_data['status'] = classified_station_data[
        'status_code'].map({
            0: "empty",
            1: "nearly empty",
            2: "OK",
            3: "nearly full",
            4: 'full'
        })

    return classified_station_data
