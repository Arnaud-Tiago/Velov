import streamlit as st
import pandas as pd

import pydeck as pdk

import utils


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
                           utils.get_stations_info().capacity.values[0]),
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


## Plotting the map

# Creating the initial view (position, zoom and pitch)
view_state = pdk.ViewState(
            latitude=45.7640,
            longitude=4.8357,
            zoom=13,
            pitch=40,
        )

# Creating the classification layers for the stations
layers = []

for status,color in STATUS_COLOURS.items():
    layers.append(
        create_scatter_layer(
            data=get_sub_dataframe_from_status(stations_and_classification, status),
            color=COLOURS.get(color)))

stations_and_classification['text_to_display'] = stations_and_classification.apply(lambda row: f"{row['bikes']}/{row['capacity']}", axis=1)

print(stations_and_classification.head())

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

# Plotting the layers on the chart
st.pydeck_chart(
    pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
    ))

pred_horizon = st.sidebar.slider(label="Prediction horizon",
                                 min_value=0,
                                 max_value=60,
                                 step=5)








#st.map(stations_info[['lat', 'lng']].rename(columns={'lng': 'lon'}))
#live_status = utils.get_live_status()

#chart_data = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] +
#                          [37.76, -122.4],
#                          columns=['lat', 'lon'])

# hexa_layer_1 = pdk.Layer(
#     'HexagonLayer_1',
#     data=stations_info[['lng', 'lat']].rename(columns={'lng': 'lon'})[:100],
#     get_position=['lon', 'lat'],
#     get_elevation=['bikes'],
#     radius=20,
#     elevation_scale=1,
#     elevation_range=[0, 1000],
#     pickable=True,
#     extruded=True,
# )

# hexa_layer_2 = pdk.Layer(
#     'HexagonLayer_2',
#     data=stations_info[['lng', 'lat']].rename(columns={'lng': 'lon'})[100:],
#     get_position=['lon', 'lat'],
#     get_elevation=['bikes'],
#     radius=20,
#     elevation_scale=1,
#     elevation_range=[0, 1000],
#     pickable=True,
#     extruded=True,
#     get_fill_color=[180, 0, 200, 140]
# )

# fig = px.scatter_geo(
#     data_frame=stations_info,
#     #color="color_column",
#     lon="lng",
#     lat="lat",
#     projection="natural earth",
#     scope='europe',
#     #hover_name="hover_column",
#     #size="size_column",  # <-- Set the column name for size
#     #height=800,
# )

#st.plotly_chart(fig, use_container_width=True)

# ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c4/Projet_bi%C3%A8re_logo_v2.png"

# icon_data = {
#     # Icon from Wikimedia, used the Creative Commons Attribution-Share Alike 3.0
#     # Unported, 2.5 Generic, 2.0 Generic and 1.0 Generic licenses
#     "url": ICON_URL,
#     "width": 242,
#     "height": 242,
#     "anchorY": 242,
# }

# icon_layer = pdk.Layer(
#     type="IconLayer",
#     data=stations_info[['station_number','lng', 'lat']],
#     get_icon="icon_data",
#     get_size=4,
#     size_scale=15,
#     get_position=["lon", "lat"],
#     pickable=True,
# )
