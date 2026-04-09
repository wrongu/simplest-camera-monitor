FROM hdgigante/python-opencv:4.13.0-alpine

COPY requirements.txt .
RUN apk add g++ cmake
RUN pip install --no-cache-dir -r requirements.txt
RUN apk del g++ cmake

COPY app /app
WORKDIR /app

CMD ["python", "main.py"]
