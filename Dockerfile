FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .
# Supabase's TLS cert is signed by Supabase's own CA, which is not in the
# OS trust store, so the pooler connection needs this file to verify it.
# Without it the app cannot connect at all.
COPY certs ./certs

# The app refuses to boot without SESSION_SECRET_KEY, so a misconfigured deploy
# fails fast and visibly instead of silently signing cookies with a default.
ENV SESSION_HTTPS_ONLY=1

# Every secret is injected by the platform, never baked into a layer:
#   Render: Dashboard -> Environment, or `render.yaml` with sync: false
#   Fly:    fly secrets set DATABASE_URL='...'
# DATABASE_URL, SESSION_SECRET_KEY, LLM_API_KEY

EXPOSE 8080

# Shell form so $PORT expands. Render injects PORT (10000 on free); Fly sets
# 8080. The app runs `alembic upgrade head` on boot under an advisory lock, so
# no separate migration step is needed on either platform.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log
