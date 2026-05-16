from collections import defaultdict
import logging
import time

ONE_HOUR_SECONDS = 3600
LogLevel = int


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class LogHandler(logging.Handler):
    def __init__(self, batch_every: float = ONE_HOUR_SECONDS):
        super().__init__()
        self.batch_every = batch_every
        self.last_emit_time = defaultdict(float)
        self.message_count = defaultdict(int)

    def emit(self, record: logging.LogRecord) -> None:
        key = repr(record.msg)
        self.message_count[key] += 1
        now = time.time()
        if now - self.last_emit_time[key] >= self.batch_every:
            count = self.message_count[key]
            if count > 1:
                record.msg = f"{record.msg} (repeated {count} times)"
            print(record.getMessage())
            self.last_emit_time[key] = now
            self.message_count[key] = 0


def get_logger(name: str, level=logging.INFO, batching: int = ONE_HOUR_SECONDS) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(LogHandler(batching))
    return logger


__all__ = [
    "LogHandler",
    "LogLevel",
    "get_logger",
]
