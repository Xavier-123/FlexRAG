from __future__ import annotations

import logging

_INITIALIZED = False

def setup_logging(log_level: str = "INFO", log_format: str | None = None) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return  # 防止重复初始化

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = log_format or "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

    root = logging.getLogger()

    # 清理已有 handler（防止重复打印）
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    root.setLevel(level)
    root.addHandler(handler)

    _INITIALIZED = True


if __name__ == '__main__':
    setup_logging()
    logging.info("Logging configured successfully.")
