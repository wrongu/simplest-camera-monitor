# If we use base-python:3.13, then `apk add py3-whatever` packages show up in `/usr/lib/python3.12/site-packages/` (version mismatch, unusable)
# If we use base-python:3.14, then `apk add py3-whatever` packages show up in `/usr/lib/python3.13/site-packages/` (version mismatch, unusable)
# So we use base-python:3.12, which is the latest version that `apk add py3-whatever` packages show up in `/usr/lib/python3.12/site-packages/` (version match, usable, but requires PYTHONPATH update)
FROM ghcr.io/home-assistant/base-python:3.12-alpine3.23

# Use `apk add` to install packages which don't build easily with pip. This is much faster than
# building them from source. The downside is that we get less control over package versions
# (like opencv-python vs opencv-contrib-python-headless)
RUN apk add py3-scikit-learn py3-opencv
# For some reason, apk-installed packages show up in /usr/lib/python3.12/site-packages/
# but the base-python:3.12 image expects them to be in /usr/local/lib/python3.12/site-packages/
RUN mv /usr/lib/python3.12/site-packages/* /usr/local/lib/python3.12/site-packages/

COPY app /app
WORKDIR /app

COPY requirements.txt .
RUN apk add g++ cmake
RUN pip install --no-cache-dir -r requirements.txt
RUN apk del g++ cmake

CMD ["python", "main.py"]
