#save data from cloud
import os
import pandas as pd
from google.cloud import storage
import datetime

from params import LOCAL_DATA_PATH_CLEAN, LOCAL_DATA_PATH_RAW, LOCAL_ROOT_PATH, BUCKET_NAME

root_path = LOCAL_ROOT_PATH
raw_data_path = LOCAL_DATA_PATH_RAW
cleaned_data_path = LOCAL_DATA_PATH_CLEAN


def save_data(df:pd.DataFrame, table_name:str, clean:bool=False ,destination='local'):
    """
    Saves a DataFrame df
        * name it with the table_name argument
        * store it in raw folder if clean is False, else store it in clean Folder
        * destination may be 'local' or 'cloud'
    
    """
    
    valid = {'local','cloud'}
    if destination not in valid:
        raise ValueError("Error : destination must be one of %r." % valid)
    
    if destination=="cloud":
        client = storage.Client() # use service account credentials
        export_bucket = client.get_bucket(BUCKET_NAME) #define bucket
        if clean:
            blob_name = 'cleaned/'+table_name
        else :
            blob_name = 'raw/'+table_name        
        # HARD STOP
        export_bucket.blob(blob_name).upload_from_string(df.to_csv(),"text/csv")
        print(f"DataFrame successfully saved on the cloud at {BUCKET_NAME}/{blob_name}")


    else:
        if clean :
            path = cleaned_data_path+table_name+'.csv'
        else :
            path = raw_data_path+table_name+'.csv'
        df.to_csv(path)
        print(f"DataFrame successfully saved locally on {path}")  
    
    return None


def load_data(table_name:str, clean:bool=False, provenance='local') -> pd.DataFrame:
    """
    input :
        * table_name as a str
        * clean as a boolean : determines if the table is to be load from cleaned or raw folder
        * provenance : can be 'local' or 'cloud' 
    ------
    returns : a pd.DataFrance
    """    
    valid = {'local','cloud'}
    if provenance not in valid:
        raise ValueError("Error : destination must be one of %r." % valid)
    
    if provenance == 'local':
        if clean : 
            path = cleaned_data_path+table_name+'.csv'
        else :
            path = raw_data_path+table_name+'.csv'
        df = pd.read_csv(path)
        print(f"DataFrame successfully loaded from {path}.")

    else :
        client = storage.Client() # use service account credentials
        bucket = client.bucket(BUCKET_NAME)
        if clean : 
            blob_name = 'cleaned/'+table_name
        else :
            blob_name = 'raw/'+table_name   
        df = pd.read_csv('gs://'+BUCKET_NAME+'/'+blob_name)     
        print(f"DataFrame successfully loaded from {BUCKET_NAME}/{blob_name}.")
    
    return df
