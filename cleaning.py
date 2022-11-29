import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from utils import get_station, to_minute

def clean_station(number:int, start_date:datetime.date='2000-01-01', step:int=5) -> pd.DataFrame:
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
