import streamlit as st
import pandas as pd

import pydeck as pdk

import utils
from app_utils import get_sub_dataframe_from_status,create_scatter_layer,classify_station


COLOURS = {
    'Black': [0, 0, 0, 150],
    'White': [255,255,255,150],
    'Gold': [210, 160, 30, 150],
    'Grey':[128,128,128,150],
    'Red': [189, 27, 33, 150],
    'Orange': [255, 165, 0, 150],
    'Green': [34, 139, 34, 150],
    'Green blue':[8,143,143,150],
    'Dark blue':[0,71,171,150]
}

STATUS_COLOURS = {
    'empty' : 'Red',
    'nearly empty': 'Orange',
    'OK':'Green',
    'nearly full': 'Green blue',
    'full':'Dark blue'
    }

st.write('Hello boys!')
st.write('Welcome to the Velo\'V prediction app')

## Pre-treatment

# Static data extraction (locally)
stations_info = utils.get_stations_info(source='local')
stations_info = stations_info[(stations_info['lng'] != 0)
                              & (stations_info['lng'] != 0)]

# Merging with live status
live_status = utils.get_live_status()
live_status['station_number'] = live_status['station_number'].astype('int64', errors='raise')
cols_to_keep = ['station_number', 'lat', 'lng', 'capacity']
stations_and_status = pd.merge(stations_info[cols_to_keep],
                               live_status,
                               how='left',
                               on='station_number').dropna()

# Creating classification
stations_and_classification = classify_station(stations_and_status)


## Creating layers
# Creating the classification layers for the stations
layers = []

for status,color in STATUS_COLOURS.items():
    layers.append(
        create_scatter_layer(
            data=get_sub_dataframe_from_status(stations_and_classification, status),
            color=COLOURS.get(color)))

# Creating a text layer with number of bikes over capacity
stations_and_classification['text_to_display'] = stations_and_classification.apply(
    lambda row: f"{row['bikes']}/{row['capacity']}", axis=1
    )


text_layer = pdk.Layer("TextLayer",
                       data=stations_and_classification,
                       pickable=True,
                       get_position=["lng", "lat"],
                       get_text='text_to_display',
                       get_color=COLOURS.get('White'),
                       billboard=False,
                       get_size=30,
                       sizeUnits='meters',
                       get_angle=0,
                       get_text_anchor='"middle"',
                       get_alignment_baseline="'center'")

layers.append(text_layer)

## Plotting the map

# Creating the initial view (position, zoom and pitch)
view_state = pdk.ViewState(
    latitude=45.7640,
    longitude=4.8357,
    zoom=13,
    pitch=40,
)

# Plotting the layers on the chart
st.pydeck_chart(
    pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
    ))

pred_horizon = st.slider(label="Prediction horizon",
                                 min_value=0,
                                 max_value=60,
                                 step=5)
