import logging
import os
from functools import lru_cache


class _Logger(logging.Logger):
    r"""A logger that supports rank0 logging."""

    def info_rank0(self, *args, **kwargs) -> None:
        self.info(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)

    def warning_rank0_once(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)


def get_logger(name: str | None = None) -> "_Logger":
    return logging.getLogger(name)  # type: ignore


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_rank0_once = warning_rank0_once


def set_global_logger(log_level: int = logging.INFO):
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [%(levelname)s|Rank{os.environ.get('LOCAL_RANK', '?')}|%(name)s:%(lineno)s] >> %(message)s",
        handlers=[logging.StreamHandler()],
    )
