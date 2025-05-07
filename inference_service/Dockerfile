FROM python:3.10-slim

LABEL maintainer="Grishma Deshmukh <gd2574@nyu.edu>"

WORKDIR /app

COPY app.py .
COPY requirements.txt .
COPY templates/ templates/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
