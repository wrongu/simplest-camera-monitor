from collections import defaultdict
import logging
import time

ONE_HOUR_SECONDS = 3600
LogLevel = int


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
            super().emit(record)
            self.last_emit_time[key] = now
            self.message_count[key] = 0


__all__ = [
    "LogHandler", 
    "LogLevel"
]