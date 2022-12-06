import pandas as pd
import pydeck as pdk
import requests

def create_scatter_layer(data, color, radius):
    '''
    Takes a DataFrame with 'lng' and 'lat' columns
    Create a ScatterLayer as a pdk.Layer object
    '''
    return pdk.Layer('ScatterplotLayer',
                     data=data,
                     get_position=['lng', 'lat'],
                     get_radius=radius,
                     size_scale=15,
                     get_fill_color=color)

def create_text_layer(data,color,size):
    '''
    Takes a DataFrame with 'lng', 'lat' and 'text_to_display' columns
    Create a TextLayer as a pdk.Layer object
    '''
    return pdk.Layer("TextLayer",
                           data=data,
                           pickable=True,
                           get_position=["lng", "lat"],
                           get_text='text_to_display',
                           get_color=color,
                           billboard=False,
                           get_size=size,
                           sizeUnits='meters',
                           get_angle=0,
                           get_text_anchor='"middle"',
                           get_alignment_baseline="'center'")

def create_pin_layer(lat, lon, size):
    '''
    Takes latitude, longitude and size
    Returns an IconLayer as a pdk.Layer object
    '''
    ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/f/fb/Map_pin_icon_green.svg"

    icon_data = {
        # Icon from Wikimedia, used the Creative Commons Attribution-Share Alike 3.0
        # Unported, 2.5 Generic, 2.0 Generic and 1.0 Generic licenses
        "url": ICON_URL,
        "width": 94,
        "height": 128,
        "anchorY": 242,
    }

    icon_layer = pdk.Layer(
        type="IconLayer",
        data=pd.DataFrame([lat, lon, icon_data],
                          index=['lat', 'lon', 'icon_data']).transpose(),
        get_icon="icon_data",
        get_size=size,
        size_scale=15,
        get_position=["lon", "lat"],
        pickable=True,
    )
    return icon_layer


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


def geocode(address):
    '''
    Takes an address. Return address attributesas a dictionary.
    Keys are including "display_name", "lat" and "lon"
    '''
    params_geo = {"q": address, 'format': 'json'}
    places = requests.get(f"https://nominatim.openstreetmap.org/search",
                          params=params_geo).json()
    return places[0]
