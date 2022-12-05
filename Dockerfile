FROM python:3.10.6
COPY requirements.txt /requirements.txt

# COPY model.pkl /model.pkl
# COPY app.py /app.py
COPY utils.py /utils.py
COPY fast.py /fast.py
# COPY cleaning.py /cleaning.py
# COPY data.py /data.py
# COPY main_interface.py /main_interface.py
# COPY model.py /model.py
COPY params.py /params.py
# COPY setup.py /setup.py
# COPY app_utils.py /app_utils.py
COPY dummy_random_model.keras /dummy_random_model.keras
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip freeze > requirements.txt
#CMD uvicorn app.simple:app --host 0.0.0.0
CMD uvicorn fast:app --host 0.0.0.0 --port 8000
#CMD uvicorn app.main:app --host 0.0.0.0 --port 8000


#FROM tensorflow/tensorflow:2.10.0

#COPY requirements_prod.txt requirements.txt





#CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
