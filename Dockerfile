FROM python:3.10.6-buster

COPY fakenews /fakenews
COPY credentials.json /credentials.json
COPY requirements.txt /requirements.txt
COPY .env /.env

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt
CMD uvicorn fakenews.api.fast:app --host 0.0.0.0 --port $PORT
