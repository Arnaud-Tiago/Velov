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


run_train:


run_pred:


run_evaluate:


run_all: run_preprocess run_train run_pred run_evaluate

# legacy directive
run_model: run_all
