FROM python:3.10.6

COPY requirements.txt /requirements.txt
COPY utils.py /utils.py
COPY fast.py /fast.py
COPY params.py /params.py
COPY dummy_random_model_3.pkl /dummy_random_model_3.pkl

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn fast:app --host 0.0.0.0 --port $PORT

# COPY model.pkl /model.pkl
# COPY app.py /app.py
# COPY cleaning.py /cleaning.py
# COPY data.py /data.py
# COPY main_interface.py /main_interface.py
# COPY setup.py /setup.py
# COPY app_utils.py /app_utils.py
#CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
#COPY requirements_prod.txt requirements.txt
