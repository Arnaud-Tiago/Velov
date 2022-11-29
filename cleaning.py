import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from utils import get_station, to_minute

def clean_station(number:int, start_date:datetime.date='2000-01-01') -> pd.DataFrame:
    """
    input : a station number and a start date
    ---
    returns : a data frame with data every 5 minutes 
    """
    df = get_station(number, start_date)
    return df.resample('5min', on='time').mean().fillna(method='ffill').drop(columns='index')
