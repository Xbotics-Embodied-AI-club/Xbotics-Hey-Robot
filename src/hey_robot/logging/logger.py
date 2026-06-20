"""Hey Robot 日志核心功能。

生产级日志，支持结构化高亮与主题。
本模块提供带格式化的控制台日志输出。
"""

import importlib
import io
import json
import logging
import math
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from hey_robot.logging.styles import COLORS, FORMATS

# On Windows, sys.stdout defaults to the system locale encoding (e.g. cp936).
# Force UTF-8 so that Chinese log messages survive output capture (uv run, CI, etc.).
_stream = sys.stdout
if hasattr(_stream, "buffer"):
    try:
        _stream = io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        _stream = sys.stdout


def _get_clock_emoji(t: float) -> str:
    """根据耗时返回旋转的时钟 emoji。"""
    clocks = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]
    idx = round(t * 2) % len(clocks)
    return clocks[idx]


class _TimeElapser:
    """带实时更新的计时器上下文管理器。"""

    def __init__(self, logger, refresh_rate: int = 10, end_msg: str = ""):
        self.logger = logger
        self.dt = 1.0 / refresh_rate
        self.n = max(1, math.ceil(math.log10(refresh_rate)))
        self._stop = threading.Event()

        self.last_logger_output = self.logger.last_output
        self.start_msg = self.last_logger_output[:-4]
        self.end_msg = end_msg + self.last_logger_output[-4:] + "\n"

    def __enter__(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self.thread.join()

    def run(self):
        """在后台线程运行计时器更新循环。"""
        self.logger.raw("\x1b[1F" + self.start_msg + " ")
        t_start = time.perf_counter()
        t_elapsed = time.perf_counter() - t_start
        self.logger.raw(f"~<{t_elapsed:.{self.n}f}s>~ {_get_clock_emoji(t_elapsed)} ")
        prev_width = len(f"{t_elapsed:.{self.n}f}s ") + 3

        while not self._stop.is_set():
            time.sleep(self.dt)
            with self.logger.lock_timer():
                t_elapsed = time.perf_counter() - t_start

                if self.logger._is_new_line:
                    self.logger.raw(self.start_msg + " ")
                else:
                    self.logger.raw("\b" * prev_width)

                self.logger.raw(
                    f"~<{t_elapsed:.{self.n}f}s>~ {_get_clock_emoji(t_elapsed)} "
                )
                prev_width = len(f"{t_elapsed:.{self.n}f}s ") + 3

        self.logger.raw("\b\b\b✅ " + self.end_msg)


def _resolve_timezone(timezone_name: str | None):
    normalized = (timezone_name or "").strip()
    if not normalized or normalized.lower() in {"local", "system"}:
        local_dt = datetime.now().astimezone()
        return local_dt.tzinfo, local_dt.tzname() or "local"
    if normalized.upper() == "UTC":
        return UTC, "UTC"
    try:
        zoneinfo_module = importlib.import_module("zoneinfo")
    except ImportError:  # pragma: no cover
        zoneinfo_module = None
    if zoneinfo_module is not None:
        try:
            return zoneinfo_module.ZoneInfo(normalized), normalized
        except zoneinfo_module.ZoneInfoNotFoundError:
            local_dt = datetime.now().astimezone()
            return local_dt.tzinfo, local_dt.tzname() or "local"
    local_dt = datetime.now().astimezone()
    return local_dt.tzinfo, local_dt.tzname() or "local"


_log_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "log_context", default=None
)


def set_log_context(**kwargs: Any) -> Token:
    """设置结构化日志上下文（trace_id, episode_id 等）。

    返回一个 token，调用 reset_log_context(token) 可恢复。
    """
    return _log_context.set(kwargs)


def reset_log_context(token: Token) -> None:
    """恢复日志上下文。"""
    _log_context.reset(token)


class HeyRobotJsonFormatter(logging.Formatter):
    """JSON 行格式化器，用于结构化日志输出。"""

    def __init__(self, timezone_name: str | None = "Asia/Shanghai"):
        super().__init__()
        self._tzinfo, self._tzname = _resolve_timezone(timezone_name)

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=self._tzinfo).isoformat()
        ctx = _log_context.get() or {}
        data: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "module": getattr(record, "module_tag", "") or "",
            "msg": record.getMessage(),
        }
        for key in ("trace_id", "episode_id", "agent_id", "robot_id"):
            val = ctx.get(key)
            if val:
                data[key] = val
        return json.dumps(data, ensure_ascii=False)


class HeyRobotFormatter(logging.Formatter):
    """Hey Robot 自定义 formatter，支持结构化高亮。

    支持标记：
        ~<text>~       -> MINT 色
        ~~<text>~~     -> MINT + UNDERLINE
        ~~~<text>~~~   -> MINT + ITALIC
        ~~~~<text>~~~~ -> MINT + BOLD + ITALIC

    参数:
        verbose_time: 为 True 时显示毫秒级时间戳
        theme: 颜色主题（"dark", "light", "dumb"）
    """

    def __init__(
        self,
        verbose_time: bool = False,
        theme: str = "dark",
        timezone_name: str | None = "Asia/Shanghai",
    ):
        super().__init__()

        self.colors = COLORS(theme)
        self.formats = FORMATS(theme)
        self._tzinfo, self.timezone_name = _resolve_timezone(timezone_name)

        self.level_colors = {
            logging.DEBUG: self.colors.green,
            logging.INFO: self.colors.blue,
            logging.WARNING: self.colors.yellow,
            logging.ERROR: self.colors.red,
            logging.CRITICAL: self.colors.red,
        }

        if verbose_time:
            self.time_fmt = "%(asctime)s.%(msecs)03d"
            self.date_fmt = "%y-%m-%d %H:%M:%S"
            self.info_length = 44
        else:
            self.time_fmt = "%(asctime)s"
            self.date_fmt = "%H:%M:%S"
            self.info_length = 31

        self.last_output = ""
        self.last_color = ""

    def _time_tuple(self, created: float | None) -> time.struct_time:
        timestamp = 0.0 if created is None else created
        return datetime.fromtimestamp(timestamp, tz=self._tzinfo).timetuple()

    def colored_fmt(self, color: str) -> str:
        """生成带颜色的格式串。

        参数:
            color: ANSI 颜色码

        返回:
            格式化模板字符串
        """
        self.last_color = color
        return f"{color}[HeyRobot] [{self.time_fmt}] [%(levelname)s] %(message)s{self.formats.reset}"

    def extra_fmt(self, msg: str) -> str:
        """应用结构化高亮标记。

        参数:
            msg: 含标记的消息

        返回:
            带 ANSI 颜色码的消息
        """
        msg = msg.replace(
            "~~~~<", self.colors.mint + self.formats.bold + self.formats.italic
        )
        msg = msg.replace("~~~<", self.colors.mint + self.formats.italic)
        msg = msg.replace("~~<", self.colors.mint + self.formats.underline)
        msg = msg.replace("~<", self.colors.mint)

        msg = msg.replace(">~~~~", self.formats.reset + self.last_color)
        msg = msg.replace(">~~~", self.formats.reset + self.last_color)
        msg = msg.replace(">~~", self.formats.reset + self.last_color)
        msg = msg.replace(">~", self.formats.reset + self.last_color)

        return msg

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录。

        参数:
            record: 日志记录

        返回:
            格式化后的日志文本
        """
        log_fmt = self.colored_fmt(
            self.level_colors.get(record.levelno, self.colors.blue)
        )
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        formatter.converter = self._time_tuple
        msg = self.extra_fmt(formatter.format(record))
        self.last_output = msg
        return msg


class HeyRobotLogger:
    """Hey Robot 生产级 Logger。

    特性：
        - 主题化彩色输出
        - 结构化高亮（标记语法）
        - 线程安全的计时器输出
        - 进度条的原始输出支持
        - 模块标签显示

    参数:
        name: 模块名称，显示在日志中（如 "gateway"、"voice"）
        level: 日志等级（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        verbose_time: 为 True 时显示毫秒级时间戳
        theme: 颜色主题（"dark", "light", "dumb"）
        timezone: 时区名称
        file_path: 可选文件日志路径，启用后日志同时写入文件
    """

    def __init__(
        self,
        name: str | None = None,
        level: str = "INFO",
        verbose_time: bool = False,
        theme: str = "dark",
        timezone: str | None = "Asia/Shanghai",
        file_path: str | None = None,
        json_format: bool = False,
        json_file_path: str | None = None,
        throttle_sec: float = 0.0,
    ):
        self._name = name
        self._throttle_sec = max(0.0, throttle_sec)
        self._throttle_cache: dict[tuple[int, str], float] = {}
        self._throttle_lock = threading.Lock()

        if isinstance(level, str):
            level = level.upper()

        self._logger = logging.getLogger("hey_robot")
        self._logger.setLevel(getattr(logging, level))
        self._logger.propagate = False

        self._formatter: HeyRobotFormatter = HeyRobotFormatter(
            verbose_time, theme, timezone
        )

        existing_handlers = [
            handler
            for handler in self._logger.handlers
            if isinstance(handler, logging.StreamHandler)
        ]
        if existing_handlers:
            self._handler = existing_handlers[0]
            self._handler.setLevel(getattr(logging, level))
            self._handler.setFormatter(self._formatter)
        else:
            self._handler = logging.StreamHandler(_stream)
            self._handler.setLevel(getattr(logging, level))
            self._handler.setFormatter(self._formatter)
            self._logger.addHandler(self._handler)

        self._stream = self._handler.stream
        self._is_new_line = True

        self.timer_lock = threading.Lock()

        # File handler (plain text)
        self._file_handler: RotatingFileHandler | None = None
        if file_path:
            self._add_file_handler(file_path, level)

        # JSON handlers
        self._json_file_handler: RotatingFileHandler | None = None
        self._json_stdout_handler: logging.StreamHandler | None = None
        if json_file_path:
            self._add_json_file_handler(json_file_path, level)
        if json_format:
            self._add_json_stdout_handler(level)

    @classmethod
    def from_spec(cls, spec: Any, name: str | None = None) -> "HeyRobotLogger":
        """从 LoggingSpec 或类似对象创建 logger。

        参数:
            spec: 具有 level, theme, file_path 属性的配置对象
            name: 模块名称
        """
        return cls(
            name=name,
            level=getattr(spec, "level", "INFO"),
            theme=getattr(spec, "theme", "dark"),
            file_path=getattr(spec, "file_path", None),
            json_format=getattr(spec, "json_format", False),
            json_file_path=getattr(spec, "json_file_path", None),
            throttle_sec=float(getattr(spec, "throttle_sec", 0.0) or 0.0),
        )

    def _tag_msg(self, message: str) -> str:
        """为消息添加模块标签。"""
        if self._name:
            return f"[~<{self._name}>~] {message}"
        return message

    def _add_file_handler(self, file_path: str, level: str) -> None:
        """添加 RotatingFileHandler，使用纯文本 formatter。"""
        from pathlib import Path

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        self._file_handler = RotatingFileHandler(
            file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        self._file_handler.setLevel(getattr(logging, level))

        plain_fmt = "[HeyRobot] [%(asctime)s] [%(levelname)s] %(message)s"
        self._file_handler.setFormatter(
            logging.Formatter(plain_fmt, datefmt="%y-%m-%d %H:%M:%S")
        )
        self._logger.addHandler(self._file_handler)

    def _add_json_file_handler(self, json_file_path: str, level: str) -> None:
        """添加 JSON 格式的文件日志 handler。"""
        from pathlib import Path

        Path(json_file_path).parent.mkdir(parents=True, exist_ok=True)
        self._json_file_handler = RotatingFileHandler(
            json_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        self._json_file_handler.setLevel(getattr(logging, level))
        self._json_file_handler.setFormatter(HeyRobotJsonFormatter())
        self._logger.addHandler(self._json_file_handler)

    def _add_json_stdout_handler(self, level: str) -> None:
        """添加 JSON 格式的 stdout handler（用于容器环境）。"""
        self._json_stdout_handler = logging.StreamHandler(_stream)
        self._json_stdout_handler.setLevel(getattr(logging, level))
        self._json_stdout_handler.setFormatter(HeyRobotJsonFormatter())
        self._logger.addHandler(self._json_stdout_handler)

    def _should_throttle(self, level: int, message: str) -> bool:
        """限流检查：同一消息在 throttle_sec 窗口内只输出一次。"""
        if self._throttle_sec <= 0:
            return False
        key = (level, message[:120])
        now = time.monotonic()
        with self._throttle_lock:
            last = self._throttle_cache.get(key)
            if last is not None and (now - last) < self._throttle_sec:
                return True
            self._throttle_cache[key] = now
            if len(self._throttle_cache) > 2000:
                cutoff = now - self._throttle_sec * 3
                self._throttle_cache = {
                    k: v for k, v in self._throttle_cache.items() if v > cutoff
                }
        return False

    @property
    def info_length(self) -> int:
        """获取 INFO 级别日志前缀长度。"""
        return self._formatter.info_length

    @property
    def level(self) -> int:
        """获取当前日志等级。"""
        return self._logger.level

    @property
    def timezone_name(self) -> str:
        return str(self._formatter.timezone_name)

    @property
    def last_output(self) -> str:
        """获取最后一次格式化输出。"""
        return self._formatter.last_output

    @property
    def handler(self):
        """获取 stream handler。"""
        return self._handler

    @contextmanager
    def log_wrapper(self):
        """线程安全日志的上下文管理器。

        确保计时器输出不干扰日志消息。
        """
        self.timer_lock.acquire()

        if not self._is_new_line and not self._stream.closed:
            self._stream.write("\r")
        try:
            yield
        finally:
            self._is_new_line = True
            self.timer_lock.release()

    @contextmanager
    def lock_timer(self):
        """锁住计时器输出，保证线程安全。"""
        self.timer_lock.acquire()
        try:
            yield
        finally:
            self.timer_lock.release()

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any):
        """按指定日志等级输出消息。

        参数:
            level: 日志等级
            msg: 日志消息
            *args: 格式化位置参数
            **kwargs: 格式化关键字参数
        """
        with self.log_wrapper():
            self._logger.log(level, msg, *args, **kwargs)

    def debug(self, message: str):
        """输出 debug 消息。"""
        level = logging.DEBUG
        if self._should_throttle(level, message):
            return
        with self.log_wrapper():
            self._logger.debug(
                self._tag_msg(message), extra={"module_tag": self._name or ""}
            )

    def info(self, message: str):
        """输出 info 消息。"""
        level = logging.INFO
        if self._should_throttle(level, message):
            return
        with self.log_wrapper():
            self._logger.info(
                self._tag_msg(message), extra={"module_tag": self._name or ""}
            )

    def warning(self, message: str):
        """输出 warning 消息。"""
        level = logging.WARNING
        if self._should_throttle(level, message):
            return
        with self.log_wrapper():
            self._logger.warning(
                self._tag_msg(message), extra={"module_tag": self._name or ""}
            )

    def error(self, message: str):
        """输出 error 消息。"""
        with self.log_wrapper():
            self._logger.error(
                self._tag_msg(message), extra={"module_tag": self._name or ""}
            )

    def critical(self, message: str):
        """输出 critical 消息。"""
        with self.log_wrapper():
            self._logger.critical(
                self._tag_msg(message), extra={"module_tag": self._name or ""}
            )

    def exception(self, message: str):
        """输出 error 消息并附带异常堆栈。"""
        with self.log_wrapper():
            self._logger.exception(
                self._tag_msg(message), extra={"module_tag": self._name or ""}
            )

    def raw(self, message: str):
        """输出原始消息（不格式化）。

        用于进度条与计时器。
        """
        self._stream.write(self._formatter.extra_fmt(message))
        self._stream.flush()
        if message.endswith("\n"):
            self._is_new_line = True
        else:
            self._is_new_line = False

    def timer(self, msg: str, refresh_rate: int = 10, end_msg: str = ""):
        """创建计时器上下文管理器。

        参数:
            msg: 计时器提示消息
            refresh_rate: 刷新频率（Hz）
            end_msg: 计时结束提示

        返回:
            TimeElapser 上下文管理器
        """
        self.info(msg)
        return _TimeElapser(self, refresh_rate, end_msg)

    def add_filter(self, log_filter):
        """添加日志过滤器。"""
        self._logger.addFilter(log_filter)

    def remove_filter(self, log_filter):
        """移除日志过滤器。"""
        self._logger.removeFilter(log_filter)

    def remove_handler(self, handler):
        """移除日志 handler。"""
        self._logger.removeHandler(handler)

    def log_dict(self, metrics: dict, step: int = 0, mode: str = "train"):
        """记录指标字典。

        这是结构化指标的便捷输出方式，会格式化为日志文本。
        """
        # 格式化指标为可读字符串
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.info(f"[{mode}] step {step} | {metrics_str}")

    def close(self):
        """关闭 logger 并清理 handler。"""
        if hasattr(self, "_handler") and self._handler:
            self._handler.flush()
            self._handler.close()
        if hasattr(self, "_file_handler") and self._file_handler:
            self._file_handler.flush()
            self._file_handler.close()
        if hasattr(self, "_json_file_handler") and self._json_file_handler:
            self._json_file_handler.flush()
            self._json_file_handler.close()
        if hasattr(self, "_json_stdout_handler") and self._json_stdout_handler:
            self._json_stdout_handler.flush()
            self._json_stdout_handler.close()
        if hasattr(self, "_logger") and self._logger:
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)
