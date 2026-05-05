FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src src
COPY streamlit_app streamlit_app

ENV PYTHONPATH=/app/src:/app
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
