#save data from cloud
import os
import pandas as pd
from google.cloud import storage
import datetime

df=pd.read_csv("~/.velov/data/raw/station1.csv")

#save new blob in bucket
def save_data(df):
    if os.environ.get("DATA_SOURCE_LOAD")=="big-query":
        client = storage.Client() # use service account credentials
        bucket_name=os.environ.get("BUCKET_NAME_RAW")
        export_bucket = client.get_bucket(bucket_name) #define bucket
        blob_name=os.environ.get("BLOB_NAME")
        date=format(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
        export_bucket.blob(f"{blob_name} - {date}").upload_from_string(df.to_csv(),"text/csv")
        print("new blob created")
        return None

    #save in local
    else:
        df_local_name=os.environ.get("DATASET_LOCAL_NAME")
        df_local_path=os.environ.get("LOCAL_DATA_PATH_CLEAN")
        path=os.path.join(f"{df_local_path}",f"{df_local_name}")
        print("local saved")
        return df.to_csv(f"{path}.csv")

#download blob
def download_blob():

    if os.environ.get("DATA_SOURCE_LOAD")=="big-query":
        client = storage.Client() # use service account credentials
        bucket_name=os.environ.get("BUCKET_NAME_RAW")
        bucket = client.bucket(bucket_name) #define bucket
        blob_name=os.environ.get("BLOB_NAME")
        #date=format(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
        bucket.blob(f"{blob_name}").download_to_file("prova1.csv")
        print("done")
    return None
    '''
    BUCKET_NAME = "raw/velov_bucket"
    storage_filename = "velov-blob1 - 2022-11-30 12_42_48"
    local_filename = "prova1.csv"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)
    print("done")'''


if __name__=="__main__":
    download_blob()
