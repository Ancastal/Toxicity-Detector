FROM python:3.10.6-slim

COPY api api
COPY models models
COPY requirements.txt /requirements.txt
COPY .env /.env

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
