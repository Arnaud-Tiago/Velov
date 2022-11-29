# Save, load and clean data
import os
import pandas as pd
from google.cloud import bigquery

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """
    # remove useless/redundant columns

    # remove elements not usefull or rapresentative

#load data
def load_data():
    if os.environ.get("DATA_SOURCE")=="big-query":
        cloud_project_name=os.environ.get("PROJECT")
        cloud_dataset_name=os.environ.get("DATASET_CLOUD_NAME")
        cloud_table_name=os.environ.get("CLOUD_TABLE")
        table = f"{cloud_project_name}.{cloud_dataset_name}.{cloud_table_name}"
        client = bigquery.Client()
        rows = client.list_rows(table)
        df = rows.to_dataframe()

    else:
        df_local_name=os.environ.get("DATASET_LOCAL_NAME")
        df_local_path=os.environ.get("LOCAL_DATA_PATH_RAW")
        path=os.path.join(f"{df_local_path}",f"{df_local_name}")
        df=pd.read_csv(f"{path}.csv")

    return df

def save_data(data):
    if os.environ.get("DATA_SOURCE")=="big-query":
        cloud_project_name=os.environ.get("PROJECT")
        cloud_dataset_name=os.environ.get("DATASET_CLOUD_NAME")
        cloud_table_name=os.environ.get("CLOUD_TABLE")
        job_config = bigquery.LoadJobConfig( write_disposition = write_mode)
        client = bigquery.Client()
        table = f"{cloud_project_name}.{cloud_dataset_name}.{cloud_table_name}"
        job = client.load_table_from_dataframe(
            data, table, job_config=job_config)  # Make an API request.
        return job.result()  # Wait for the job to complete.

    else:
        df_local_name=os.environ.get("DATASET_LOCAL_NAME")
        df_local_path=os.environ.get("LOCAL_DATA_PATH_CLEAN")
        path=os.path.join(f"{df_local_path}",f"{df_local_name}")
        return data.to_csv(f"{path}.csv")




if __name__=="__main__":
    save_data(load_data())



    '''
    """
    return a chunk of the raw dataset from local disk or cloud storage
    """
    path = os.path.join(os.path.expanduser(LOCAL_DATA_PATH)
        os.path.expanduser(LOCAL_DATA_PATH),
        "processed" if "processed" in path else "raw",
        f"{path}.csv")

    if verbose:
        print(Fore.MAGENTA + f"Source data from {path}: {chunk_size if chunk_size is not None else 'all'} rows (from row {index})" + Style.RESET_ALL)

    try:

        df = pd.read_csv(
                path,
                skiprows=index + 1,  # skip header
                nrows=chunk_size,
                dtype=dtypes,
                header=None)  # read all rows






        return None  # end of data

    return df







        return chunk_df

    chunk_df = get_pandas_chunk(path=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                columns=columns,
                                verbose=verbose)

    return chunk_df

    print("\n✅ data cleaned")

    return df
'''
