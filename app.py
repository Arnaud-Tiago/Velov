import streamlit as st
import pandas as pd

import pydeck as pdk

from utils import get_stations_info, get_live_status
from app_utils import get_sub_dataframe_from_status, create_scatter_layer, classify_station,geocode,create_pin_layer

import requests

COLOURS = {
    'Black': [0, 0, 0, 150],
    'White': [255, 255, 255, 150],
    'Gold': [210, 160, 30, 150],
    'Grey': [128, 128, 128, 150],
    'Red': [189, 27, 33, 150],
    'Orange': [255, 165, 0, 150],
    'Green': [34, 139, 34, 150],
    'Green blue': [8, 143, 143, 150],
    'Dark blue': [0, 71, 171, 150]
}

STATUS_COLOURS = {
    'empty': 'Red',
    'nearly empty': 'Orange',
    'OK': 'Green',
    'nearly full': 'Green blue',
    'full': 'Dark blue'
}

DEFAULT_LAT = 45.7640
DEFAULT_LON = 4.8357
DEFAULT_ZOOM = 13
DEFAULT_CIRCLE_RADIUS = 50
DEFAULT_TEXT_SIZE = 30

# Default values for map show
lat = DEFAULT_LAT
lon = DEFAULT_LON
zoom = DEFAULT_ZOOM
radius = DEFAULT_CIRCLE_RADIUS
size = DEFAULT_TEXT_SIZE

MODEL_API_URL = "https://velovdock-w4chmhalca-ew.a.run.app/predict"

# The website starts here

st.title('Le Wagon Demo Day - Batch #1033')

st.header('Welcome to the Velo\'V prediction app')

st.caption('''
           Cycling for 20 minutes and having the unpleasant surprise of finding the destination station full on arrival, when it was empty 20 minutes earlier...
           \n Velo'V users experience this regularly. Le Wagon Lyon designed a solution for you. Based on usage history, you can now have access to a station status prediction using 5-minute increments and up to 1 hour.
''')


st.subheader('Let\'s find out together')

text_input = st.text_input(label='Address')

if text_input:
    try:
        address = geocode(text_input)
        address_name = ", ".join(address.get("display_name").split(', ')[:3])
        lat = float(address.get("lat"))
        lon = float(address.get("lon"))
        zoom = 15
        radius = 40
        size = 22

        st.caption(f'Let\'s ride to "{address_name}"')
    except IndexError:
        st.caption(f'Please enter a valid address')

pred_horizon = st.slider(label="Select your prediction horizon (in min)",
                         min_value=0,
                         max_value=60,
                         step=5)

## Pre-treatment

# Static data extraction (online)
stations_info = get_stations_info(source='online')

stations_info = stations_info[(stations_info['lng'] != 0)
                              & (stations_info['lng'] != 0)]

# Fetching the status to display, depending on pred horizon
if pred_horizon != 0:
    status_to_display = pd.DataFrame.from_dict(
        requests.get(
            MODEL_API_URL,
            timeout=10).json()).rename_axis('station_number').reset_index()
else:
    status_to_display = get_live_status()

# Merging station status to display with station infos
status_to_display['station_number'] = status_to_display[
    'station_number'].astype('int64', errors='raise')

column_selection = ['station_number', 'lat', 'lng', 'capacity']

stations_and_status = pd.merge(stations_info[column_selection],
                               status_to_display,
                               how='left',
                               on='station_number').dropna()

# Creating bike_stands columns if doesn't exist (given by "get_live_status", not given by the model)
if not 'bike_stands' in stations_and_status.columns:
    stations_and_status['bike_stands'] = stations_and_status[
        'capacity'] - stations_and_status['bikes']

# Creating classification
stations_and_classification = classify_station(stations_and_status)

## Creating layers
# Creating the classification layers for the stations
layers = []

for status, color in STATUS_COLOURS.items():
    layers.append(
        create_scatter_layer(data=get_sub_dataframe_from_status(
            stations_and_classification, status),
                             color=COLOURS.get(color),
                             radius=radius))

# Creating a text layer with number of bikes over capacity
stations_and_classification[
    'text_to_display'] = stations_and_classification.apply(
        lambda row: f"{row['bikes']}/{row['capacity']}", axis=1)

text_layer = pdk.Layer("TextLayer",
                       data=stations_and_classification,
                       pickable=True,
                       get_position=["lng", "lat"],
                       get_text='text_to_display',
                       get_color=COLOURS.get('White'),
                       billboard=False,
                       get_size=size,
                       sizeUnits='meters',
                       get_angle=0,
                       get_text_anchor='"middle"',
                       get_alignment_baseline="'center'")

layers.append(text_layer)

if lat != DEFAULT_LAT and lon != DEFAULT_LON :
    icon_layer = create_pin_layer(lat=lat,lon=lon,size=4)
    layers.append(icon_layer)

## Plotting the map

# Creating the initial view (position, zoom and pitch)

view_state = pdk.ViewState(
    latitude=lat,
    longitude=lon,
    zoom=zoom,
    pitch=0,
)

# Plotting the layers on the chart
st.pydeck_chart(pdk.Deck(
    initial_view_state=view_state,
    layers=layers,
))

st.header('Thanks all, and Ride on!')
