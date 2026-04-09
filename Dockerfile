FROM python:3.12-slim

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get remove libglib2.0-0

COPY app /app
WORKDIR /app

CMD ["python", "main.py"]
