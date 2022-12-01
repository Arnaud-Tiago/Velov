

LOCAL_DATA_PATH_RAW='~/.velov/data/raw/'
LOCAL_DATA_PATH_CLEAN='~/.velov/data/cleaned/'
LOCAL_ROOT_PATH='~/.velov/data/'

DATASET_LOCAL_NAME='station1'
LOCAL_REGISTRY_PATH= '~/.velov/mlops/training_outputs'
MODEL_TARGET='local'
DATA_SOURCE_SAVE='cloud'
DATA_SOURCE_LOAD="cloud"

# GCP Project
PROJECT= 'velov-pred'
REGION= 'europe-west1'

# Cloud Storage
BUCKET_NAME='velov_bucket'
#BUCKET_NAME_RAW='raw/velov_bucket'
DATASET_CLOUD_NAME='velov_test'
CLOUD_TABLE='station-test'
BLOB_NAME='velov-blob1'
BLOB_NAME_MODEL='model/velov-blob1'
BLOB_NAME_CLEANED='cleaned/velov-blob1'
# Compute Engine
INSTANCE='ssh-velov'
