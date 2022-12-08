#show env var
show_env:
	@echo "\nEnvironment variables used by the \`velov\` package loaded by \`direnv\` from your \`.env\` located at:"
	@echo ${DIRENV_DIR}
	@echo ${DATASET_LOCAL_NAME}

#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y proj_velov || :
	@pip install -e .

run_cleaning_cloud:
	python -c 'from data import increment_data; increment_data(source="cloud", save=True, verbose=1)'

run_cleaning_local:
	python -c 'from data import increment_data; increment_data(source='local', save=True, verbose=1)'

run_preprocess:
get training testing data

run_train:
fit model

get_savage:
	python -c 'from model import get_savage_train_test_split; get_savage_train_test_split(df:pd.DataFrame, train_size:int, test_size:int, input_length:int, output_length:int)'

handle_model:
	python -c 'from model import handle_model; handle_model(source='local', save=True, verbose=1)'

run_all: get_savage handle_model

run_model: run_all

run_api:
	uvicorn fast:app --reload
