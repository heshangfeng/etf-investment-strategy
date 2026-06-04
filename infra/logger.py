"""
ETF 智能投研系统 - 统一日志模块

用法：
  from infra.logger import logger
  logger.info("消息")
  logger.warning("警告")
  logger.error("错误", exc_info=True)
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """初始化日志配置：控制台 + 文件双输出"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"{datetime.now().strftime('%Y%m%d')}_run.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("etf")
    root.setLevel(level)
    if root.handlers:
        return  # 防止重复初始化

    # 文件 handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 控制台 handler（INFO 及以上）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def get_logger(name: str = __name__) -> logging.Logger:
    """获取 etf 命名空间下的子 logger"""
    return logging.getLogger(f"etf.{name}")


# 模块导入时自动初始化（幂等）
setup_logging()
