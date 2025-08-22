FROM python:3.9-slim

WORKDIR /code

COPY ./README.md ./

COPY ./package[s] ./packages

COPY ./app ./app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8001
