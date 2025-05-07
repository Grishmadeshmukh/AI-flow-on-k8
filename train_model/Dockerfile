FROM python:3.10-slim

LABEL maintainer="Grishma Deshmukh <gd2574@nyu.edu>"

WORKDIR /train_service

COPY requirements.txt .
COPY train.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "train.py", "--batch-size", "256", "--epochs", "1", "--save-model"]
