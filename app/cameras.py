import threading
import time
from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np
import requests
from onvif import ONVIFCamera

from app.image_loader import get_all_timestamped_files_sorted
from app.utils import get_logger

logger = get_logger("cameras")


class Camera(Protocol):
    @property
    def resolution(self) -> tuple[int, int]: ...
    def get_frame(self) -> tuple[float, cv.Mat | None]: ...
    def get_last_frame(self) -> tuple[float, cv.Mat | None]: ...
    def reboot(self) -> bool: ...


_RECONNECT_AFTER_FAILURES = 10
_BACKOFF_INITIAL = 0.5
_BACKOFF_MAX = 30.0


class VideoStreamBackgroundThread(threading.Thread):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.cap = cv.VideoCapture(url)
        self.frame = None
        self._frame_lock = threading.Lock()
        self.running = True
        self._consecutive_failures = 0

    def run(self):
        backoff = _BACKOFF_INITIAL
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self._frame_lock:
                    self.frame = f
                self._consecutive_failures = 0
                backoff = _BACKOFF_INITIAL
            else:
                with self._frame_lock:
                    self.frame = None
                logger.error(f"Dropped frame; delay = {backoff}s")
                self._consecutive_failures += 1
                time.sleep(min(backoff, _BACKOFF_MAX))
                backoff = min(backoff * 2, _BACKOFF_MAX)
                if self._consecutive_failures % _RECONNECT_AFTER_FAILURES == 0:
                    logger.error("Too many stream failures; attempting reconnect")
                    self.cap.release()
                    self.cap = cv.VideoCapture(self.url)

    def is_open(self):
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.cap.release()


# def dictify(obj: object):
#     if isinstance(obj, Mapping):
#         return {k_: dictify(v_) for k_, v_ in obj.items()}
#     elif isinstance(obj, list) or isinstance(obj, tuple):
#         return [dictify(v_) for v_ in obj]
#     elif hasattr(obj, "__dict__"):
#         d = obj.__dict__
#         for k, v in d.items():
#             if k == "__values__":
#                 return dictify(v)
#             else:
#                 d[k] = dictify(v)
#         return d
#     else:
#         return obj


_NO_FRAME_TIMEOUT = 10.0


def _maybe_resize(image: cv.Mat, resolution: tuple[int, int]) -> cv.Mat:
    h, w = image.shape[:2]
    if (w, h) != resolution[:2]:
        image = cv.resize(image, resolution, interpolation=cv.INTER_AREA)
    return image


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
        resolution = tuple(resolution)
        self._resolution = resolution
        self.stream = None
        self.last_frame = None
        self.last_frame_raw = None
        self.last_frame_time = time.time()  # seed so timeout starts from init
        self._init_failed = False

        try:
            self.cam = ONVIFCamera(host, port, username, password)
            self.media = self.cam.create_media_service()
            profiles = self.media.GetProfiles()
            if not profiles:
                raise ValueError("No media profiles found on camera")

            # pprint([dictify(p) for p in profiles])

            nearest_resolution_diff = 1e6
            for profile in profiles:
                # TODO - there is sometimes a 'snapshot' interface where Encoding is "jpeg". Use that when appropriate,
                #  but also beware of much lower frame rate limits. Using the jpeg interface means no background
                #  thread for streaming, which is nice if only taking infrequent snapshots.
                if profile.VideoEncoderConfiguration.Encoding != "H264":
                    continue
                res = profile.VideoEncoderConfiguration.Resolution
                res_diff = abs(resolution[0] - res.Width) + abs(resolution[1] - res.Height)
                if res_diff < nearest_resolution_diff:
                    nearest_resolution_diff = res_diff
                    self.profile = profile

            if resolution != (
                self.profile.VideoEncoderConfiguration.Resolution.Width,
                self.profile.VideoEncoderConfiguration.Resolution.Height,
            ):
                logger.warning(
                    f"Requested resolution {resolution} "
                    f"but using {self.profile.VideoEncoderConfiguration.Resolution}"
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
            self.init_capture()
        except Exception as e:
            self._init_failed = True
            self._init_error = e

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution[:2]

    def init_capture(self):
        if self.stream is not None:
            self.stream.stop()
        self.stream = VideoStreamBackgroundThread(self.rtsp_url)
        self.stream.daemon = True
        self.stream.start()

    def get_frame(self) -> tuple[float, cv.Mat]:
        if self._init_failed:
            raise ConnectionError(f"Camera init failed: {self._init_error}")
        with self.stream._frame_lock:
            now = time.time()
            frame = self.stream.frame
        if frame is not None and not np.array_equal(frame, self.last_frame_raw):
            frame = frame.copy()
            self.last_frame_raw = frame
            frame = cv.cvtColor(_maybe_resize(frame, self._resolution), cv.COLOR_BGR2RGB)
            self.last_frame = frame
            self.last_frame_time = now
        elif frame is None and now - self.last_frame_time > _NO_FRAME_TIMEOUT:
            raise ConnectionError(f"No frames received for >{_NO_FRAME_TIMEOUT}s")
        return self.last_frame_time, self.last_frame

    def get_last_frame(self) -> tuple[float, cv.Mat | None]:
        if self.last_frame is not None and (time.time() - self.last_frame_time) < 5:
            return self.last_frame_time, self.last_frame
        else:
            return time.time(), None

    def reboot(self) -> bool:
        # ONVIF does not have a standard reboot command; restart the capture thread
        self.init_capture()
        return True


class ESPHomeCameraWrapper(Camera):
    def __init__(self, url: str, resolution: tuple[int, int]):
        self.url = url
        self.last_frame = None
        self.last_frame_time = 0
        # TODO - consider using the event stream API and keeping a persistent app:camera http connection open
        self.components = {"switch/reboot": None, "switch/flash": None}
        self.poll_components()
        self._resolution = tuple(resolution)

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def get_frame(self) -> tuple[float, cv.Mat]:
        resp = requests.get(self.url, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = _maybe_resize(cv.imdecode(image_array, cv.IMREAD_COLOR_RGB), self._resolution)
            self.last_frame = frame
            self.last_frame_time = time.time()
            return self.last_frame_time, frame
        else:
            raise ConnectionError(f"Failed to get frame: {resp.status_code}")

    def get_last_frame(self) -> tuple[float, cv.Mat]:
        if self.last_frame is not None and (time.time() - self.last_frame_time) < 5:
            return self.last_frame_time, self.last_frame
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
                logger.error(f"Failed to send reboot request: {e}")
                return False
        else:
            return False


class LoggedImagePseudoCamera(Camera):
    def __init__(self, image_dir: Path, resolution: tuple[int, int]):
        self.image_dir = image_dir
        self.last_frame = None
        self.last_frame_idx = 0
        self.timestamped_images = get_all_timestamped_files_sorted(image_dir)
        self._last_frame = None
        self._resolution = resolution

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def get_frame(self) -> tuple[float, cv.Mat]:
        self.last_frame_idx += 1
        ts, im = self.timestamped_images[self.last_frame_idx]
        self._last_frame = _maybe_resize(cv.imread(str(im), cv.IMREAD_COLOR_RGB), self._resolution)
        return ts, self._last_frame

    def get_last_frame(self) -> tuple[float, cv.Mat]:
        return self.timestamped_images[self.last_frame_idx][0], self._last_frame

    def reboot(self) -> bool:
        self.last_frame_idx = 0
        self._last_frame = None
        return True
