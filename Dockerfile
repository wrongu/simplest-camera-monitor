FROM ghcr.io/home-assistant/base-python:3.13-alpine3.23

RUN apk add g++ gcc make git
RUN apk add python3-dev python3 py3-pip py3-scikit-learn
# For some reason, py3-scikit-learn shows up in /usr/lib/python3.12/site-packages/
# which is not included by default in system-site-packages. Setting the env
# variable before creating the venv seems to fix it.
ENV PYTHONPATH="/usr/lib/python3.12/site-packages/:$PYTHONPATH"

RUN apk add build-base linux-headers samurai cmake
# RUN pip install --upgrade pip setuptools wheel meson-python
RUN pip install scikit-build numpy
ARG OPENCV_PACKAGE=opencv-contrib-python-headless
RUN MAKEFLAGS="-j4" pip install $OPENCV_PACKAGE --no-build-isolation
RUN apk del build-base linux-headers samurai cmake

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
