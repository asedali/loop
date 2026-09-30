FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# SQLite file lives here — this path is where fly.toml mounts the persistent volume
ENV LAUNCHLOOP_DB_PATH=/data/launchloop.db

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
