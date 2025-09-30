import threading
import time
from typing import Protocol

import cv2 as cv
import numpy as np
import requests
from onvif import ONVIFCamera


class Camera(Protocol):
    def get_frame(self) -> cv.Mat: ...
    def get_last_frame(self) -> cv.Mat: ...
    def reboot(self) -> bool: ...


class VideoStreamBackgroundThread(threading.Thread):
    def __init__(self, url):
        super().__init__()
        self.cap = cv.VideoCapture(url)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                self.frame = f

    def stop(self):
        self.running = False
        self.cap.release()


class ONVIFCameraWrapper(Camera):
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        resolution: tuple[int, int] = (640, 480),
    ):
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._resolution = resolution
        self.cam = ONVIFCamera(host, port, username, password)
        self.media = self.cam.create_media_service()
        profiles = self.media.GetProfiles()
        if not profiles:
            raise ValueError("No media profiles found on camera")
        available_resolutions = []
        for profile in profiles:
            res = profile.VideoEncoderConfiguration.Resolution
            available_resolutions.append((res.Width, res.Height))
        try:
            i_profile = available_resolutions.index(resolution)
            self.profile = profiles[i_profile]
        except ValueError:
            raise ValueError(
                f"Requested resolution {resolution} not available. Available: {available_resolutions}"
            )
        stream = self.media.GetStreamUri(
            {
                "StreamSetup": {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                },
                "ProfileToken": self.profile.token,
            }
        )
        rtsp_url = stream.Uri
        self.rtsp_url = rtsp_url.replace("rtsp://", f"rtsp://{username}:{password}@")
        self.stream = None
        self.init_capture()
        self.last_frame = None
        self.last_frame_time = 0

    def init_capture(self):
        self.stream = VideoStreamBackgroundThread(self.rtsp_url)
        self.stream.daemon = True
        self.stream.start()

    def get_frame(self):
        frame = self.stream.frame
        if frame is not None:
            self.last_frame = frame
            self.last_frame_time = time.time()
        return frame

    def get_last_frame(self):
        if self.last_frame is not None and (time.time() - self.last_frame_time) < 5:
            return self.last_frame
        else:
            return self.get_frame()

    def reboot(self) -> bool:
        # ONVIF does not have a standard reboot command; this is a placeholder
        self.init_capture()
        return self.stream.isOpened()


class ESPHomeCameraWrapper(Camera):
    def __init__(self, url: str):
        self.url = url
        self.last_frame = None
        self.last_frame_time = 0
        # TODO - consider using the event stream API and keeping a persistent app:camera http connection open
        self.components = {"switch/reboot": None, "switch/flash": None}
        self.poll_components()

    def get_frame(self) -> cv.Mat:
        resp = requests.get(self.url, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv.imdecode(image_array, cv.IMREAD_COLOR)
            self.last_frame = frame
            self.last_frame_time = time.time()
            return frame
        else:
            raise ConnectionError(f"Failed to get frame: {resp.status_code}")

    def get_last_frame(self) -> cv.Mat:
        if self.last_frame is not None and (time.time() - self.last_frame_time) < 5:
            return self.last_frame
        else:
            return self.get_frame()

    def poll_components(self) -> None:
        for comp in self.components.keys():
            try:
                resp = requests.get(self.url + f"/{comp}/", timeout=2)
                if resp.status_code == 200:
                    self.components[comp] = resp.json()
                else:
                    self.components[comp] = None
            except requests.exceptions.RequestException:
                self.components[comp] = None

    def reboot(self) -> bool:
        if self.components.get("switch/reboot") is not None:
            try:
                resp = requests.post(self.url + "/switch/reboot/turn_on", timeout=5)
                return resp.status_code == 200
            except Exception as e:
                print(f"Failed to send reboot request: {e}")
                return False
        else:
            return False
