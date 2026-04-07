FROM hdgigante/python-opencv:4.13.0-alpine

# Use `apk add` to install packages which don't build easily with pip. This is much faster than
# building them from source. The downside is that we get less control over package versions
# (like opencv-python vs opencv-contrib-python-headless)
# RUN apk add py3-scikit-learn py3-opencv
# For some reason, apk-installed packages show up in /usr/lib/python3.12/site-packages/
# but the base-python:3.12 image expects them to be in /usr/local/lib/python3.12/site-packages/
# RUN mv /usr/lib/python3.12/site-packages/* /usr/local/lib/python3.12/site-packages/

COPY app /app
WORKDIR /app

COPY requirements.txt .
RUN apk add g++ cmake
RUN pip install --no-cache-dir -r requirements.txt
RUN apk del g++ cmake

CMD ["python", "main.py"]
