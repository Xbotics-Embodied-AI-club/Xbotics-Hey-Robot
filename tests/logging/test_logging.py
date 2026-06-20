from __future__ import annotations

import io
import json
import logging
import re
from datetime import UTC
from pathlib import Path

import pytest

from hey_robot.config.model import DeploymentConfig, LoggingSpec
from hey_robot.logging.logger import (
    HeyRobotFormatter,
    HeyRobotLogger,
    _get_clock_emoji,
    _resolve_timezone,
)
from hey_robot.logging.styles import Colors, Formats, styless


def test_styles_return_expected_values_for_themes() -> None:
    assert Colors("dark").green.startswith("\x1b[")
    assert Colors("light").blue.startswith("\x1b[")
    assert Colors("dumb").red == ""
    assert Formats("dark").bold == "\x1b[1m"
    assert Formats("dark").italic == "\x1b[3m"
    assert Formats("dark").underline == "\x1b[4m"
    assert Formats("dumb").reset == ""
    assert styless("\x1b[38;5;119mhello\x1b[0m") == "hello"


class TestColors:
    def test_dark_theme_returns_ansi_codes(self) -> None:
        c = Colors("dark")
        assert c.green.startswith("\x1b[")
        assert c.blue.startswith("\x1b[")
        assert c.yellow.startswith("\x1b[")
        assert c.red.startswith("\x1b[")
        assert c.mint.startswith("\x1b[")
        assert c.gray.startswith("\x1b[")

    def test_light_theme_returns_ansi_codes(self) -> None:
        c = Colors("light")
        assert c.green.startswith("\x1b[")
        assert c.blue.startswith("\x1b[")
        assert c.yellow.startswith("\x1b[")
        assert c.red.startswith("\x1b[")
        assert c.mint.startswith("\x1b[")
        assert c.gray.startswith("\x1b[")

    def test_dumb_theme_returns_empty_for_all_colors(self) -> None:
        c = Colors("dumb")
        assert c.green == ""
        assert c.blue == ""
        assert c.yellow == ""
        assert c.red == ""
        assert c.mint == ""
        assert c.gray == ""

    def test_unknown_theme_returns_empty_strings(self) -> None:
        c = Colors("nonexistent")
        assert c.green == ""
        assert c.red == ""
        assert c.gray == ""

    def test_default_theme_is_dark(self) -> None:
        c = Colors()
        assert c.green.startswith("\x1b[")


class TestFormats:
    def test_dark_theme_returns_ansi(self) -> None:
        f = Formats("dark")
        assert f.bold == "\x1b[1m"
        assert f.italic == "\x1b[3m"
        assert f.underline == "\x1b[4m"
        assert f.reset == "\x1b[0m"

    def test_light_theme_returns_ansi(self) -> None:
        f = Formats("light")
        assert f.bold == "\x1b[1m"
        assert f.italic == "\x1b[3m"
        assert f.underline == "\x1b[4m"
        assert f.reset == "\x1b[0m"

    def test_dumb_theme_returns_empty(self) -> None:
        f = Formats("dumb")
        assert f.bold == ""
        assert f.italic == ""
        assert f.underline == ""
        assert f.reset == ""

    def test_unknown_theme_returns_ansi(self) -> None:
        f = Formats("nonexistent")
        assert f.bold == "\x1b[1m"
        assert f.reset == "\x1b[0m"


class TestStyless:
    def test_removes_simple_color_code(self) -> None:
        assert styless("\x1b[38;5;119mhello\x1b[0m") == "hello"

    def test_removes_multiple_codes(self) -> None:
        text = "\x1b[1m\x1b[38;5;9merror\x1b[0m"
        assert styless(text) == "error"

    def test_plain_text_unchanged(self) -> None:
        assert styless("plain text") == "plain text"

    def test_empty_string(self) -> None:
        assert styless("") == ""

    def test_only_ansi_codes(self) -> None:
        assert styless("\x1b[1m\x1b[0m") == ""


def test_resolve_timezone_handles_utc_and_fallback() -> None:
    tzinfo, name = _resolve_timezone("UTC")
    assert tzinfo is UTC
    assert name == "UTC"

    fallback_tzinfo, fallback_name = _resolve_timezone("Invalid/Zone")
    assert fallback_tzinfo is not None
    assert isinstance(fallback_name, str)
    assert fallback_name


def test_formatter_formats_time_and_markup() -> None:
    formatter = HeyRobotFormatter(verbose_time=True, theme="dumb", timezone_name="UTC")
    record = logging.LogRecord(
        "hey_robot", logging.INFO, __file__, 10, "~<hello>~", (), None
    )
    record.created = 1_700_000_000.0
    formatted = formatter.format(record)

    assert "[HeyRobot]" in formatted
    assert "[INFO]" in formatted
    assert "hello" in formatted
    assert re.search(r"\[\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\]", formatted)


def test_logger_writes_raw_and_metrics_and_closes_handlers() -> None:
    logger = HeyRobotLogger(level="INFO", theme="dumb", timezone="UTC")
    stream = io.StringIO()
    logger.handler.setStream(stream)
    logger._stream = stream

    logger.raw("raw-line\n")
    logger.log_dict({"loss": 0.25, "epoch": 2}, step=3, mode="eval")

    output = stream.getvalue()
    assert "raw-line" in output
    assert "[eval] step 3 | loss: 0.2500 | epoch: 2" in output
    assert logger.info_length > 0
    assert logger.level == logging.INFO
    assert logger.timezone_name == "UTC"

    with logger.log_wrapper():
        pass
    with logger.lock_timer():
        pass

    logger.close()
    assert logger.handler not in logger._logger.handlers


def test_logger_timer_returns_context_manager() -> None:
    logger = HeyRobotLogger(level="INFO", theme="dumb", timezone="UTC")
    stream = io.StringIO()
    logger.handler.setStream(stream)
    timer = logger.timer("starting timer", refresh_rate=5, end_msg="done")
    assert timer.logger is logger
    assert timer.dt == pytest.approx(0.2)
    assert timer.n == 1
    logger.close()


def test_clock_emoji_cycles() -> None:
    assert _get_clock_emoji(0.0)
    assert _get_clock_emoji(0.5)
    assert _get_clock_emoji(1.0)


def test_exception_outputs_traceback() -> None:
    logger = HeyRobotLogger(level="INFO", theme="dumb", timezone="UTC")
    stream = io.StringIO()
    logger.handler.setStream(stream)
    logger._stream = stream

    try:
        raise ValueError("test error for traceback")
    except ValueError:
        logger.exception("something failed")

    output = stream.getvalue()
    assert "something failed" in output
    assert "ValueError" in output
    assert "test error for traceback" in output
    logger.close()


def test_file_handler_created_when_path_given(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "test.log"
    logger = HeyRobotLogger(
        level="DEBUG", theme="dumb", timezone="UTC", file_path=str(log_file)
    )

    logger.info("file handler test")
    logger.close()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "file handler test" in content
    assert "\x1b[" not in content  # no ANSI codes in file


def test_file_handler_rotates(tmp_path: Path) -> None:
    log_file = tmp_path / "rotate.log"
    logger = HeyRobotLogger(
        level="DEBUG", theme="dumb", timezone="UTC", file_path=str(log_file)
    )

    # Write enough to exceed a small threshold — just verify it doesn't crash
    for i in range(100):
        logger.info(f"line {i}" + "x" * 1000)
    logger.close()

    assert log_file.exists()


def test_from_spec_creates_logger() -> None:
    spec = LoggingSpec(level="DEBUG", theme="dumb")
    logger = HeyRobotLogger.from_spec(spec)
    assert logger.level == logging.DEBUG
    logger.close()


def test_from_spec_with_file_path(tmp_path: Path) -> None:
    log_file = tmp_path / "spec.log"
    spec = LoggingSpec(level="INFO", theme="dumb", file_path=str(log_file))
    logger = HeyRobotLogger.from_spec(spec)
    logger.info("from spec")
    logger.close()
    assert log_file.exists()
    assert "from spec" in log_file.read_text(encoding="utf-8")


def test_logging_spec_parsed_from_yaml() -> None:
    config = DeploymentConfig.from_dict(
        {
            "logging": {
                "level": "debug",
                "theme": "light",
                "file_path": "runtime/test.log",
            },
        }
    )
    assert config.logging.level == "DEBUG"
    assert config.logging.theme == "light"
    assert config.logging.file_path == "runtime/test.log"


def test_logging_spec_defaults() -> None:
    config = DeploymentConfig.from_dict({})
    assert config.logging.level == "INFO"
    assert config.logging.theme == "dark"
    assert config.logging.file_path is None
    assert config.logging.json_format is False
    assert config.logging.json_file_path is None
    assert config.logging.throttle_sec == 0.0


def test_json_formatter_outputs_json_line() -> None:
    from hey_robot.logging.logger import HeyRobotJsonFormatter

    formatter = HeyRobotJsonFormatter(timezone_name="UTC")
    record = logging.LogRecord(
        "hey_robot", logging.INFO, __file__, 10, "test message", (), None
    )
    record.created = 1_700_000_000.0
    record.module_tag = "test"

    output = formatter.format(record)

    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["module"] == "test"
    assert data["msg"] == "test message"
    assert "ts" in data


def test_json_file_handler_writes_json(tmp_path: Path) -> None:
    json_file = tmp_path / "logs" / "test.jsonl"
    logger = HeyRobotLogger(
        name="test_mod",
        level="DEBUG",
        theme="dumb",
        timezone="UTC",
        json_file_path=str(json_file),
    )

    logger.info("json log entry")
    logger.close()

    assert json_file.exists()
    content = json_file.read_text(encoding="utf-8")

    data = json.loads(content.strip().split("\n")[0])
    assert "json log entry" in data["msg"]
    assert data["module"] == "test_mod"
    assert data["level"] == "INFO"


def test_json_stdout_handler_added_when_json_format_enabled() -> None:
    logger = HeyRobotLogger(
        name="test", level="INFO", theme="dumb", timezone="UTC", json_format=True
    )
    assert logger._json_stdout_handler is not None
    logger.close()


def test_log_context_injected_into_json_output() -> None:
    from hey_robot.logging.logger import (
        HeyRobotJsonFormatter,
        reset_log_context,
        set_log_context,
    )

    token = set_log_context(
        trace_id="abc123", episode_id="ep001", agent_id="main", robot_id="r1"
    )

    formatter = HeyRobotJsonFormatter(timezone_name="UTC")
    record = logging.LogRecord(
        "hey_robot", logging.INFO, __file__, 10, "context test", (), None
    )
    record.created = 1_700_000_000.0
    record.module_tag = "agent"

    output = formatter.format(record)

    data = json.loads(output)
    assert data["trace_id"] == "abc123"
    assert data["episode_id"] == "ep001"
    assert data["agent_id"] == "main"
    assert data["robot_id"] == "r1"

    reset_log_context(token)


def test_set_log_context_reset_clears() -> None:
    from hey_robot.logging.logger import (
        HeyRobotJsonFormatter,
        reset_log_context,
        set_log_context,
    )

    token = set_log_context(trace_id="xyz")
    formatter = HeyRobotJsonFormatter(timezone_name="UTC")
    record = logging.LogRecord("hey_robot", logging.INFO, __file__, 10, "msg", (), None)
    record.created = 1_700_000_000.0

    data1 = json.loads(formatter.format(record))
    assert data1.get("trace_id") == "xyz"

    reset_log_context(token)
    data2 = json.loads(formatter.format(record))
    assert "trace_id" not in data2


def test_throttle_suppresses_duplicate_messages() -> None:
    logger = HeyRobotLogger(
        name="throttle_test",
        level="DEBUG",
        theme="dumb",
        timezone="UTC",
        throttle_sec=10.0,
    )
    stream = io.StringIO()
    logger.handler.setStream(stream)
    logger._stream = stream

    logger.info("重复消息")
    logger.info("重复消息")
    logger.info("另一条消息")

    output = stream.getvalue()
    assert output.count("重复消息") == 1
    assert "另一条消息" in output
    logger.close()


def test_throttle_disabled_by_default() -> None:
    logger = HeyRobotLogger(name="test", level="DEBUG", theme="dumb", timezone="UTC")
    stream = io.StringIO()
    logger.handler.setStream(stream)
    logger._stream = stream

    for _ in range(5):
        logger.info("same message")
    assert stream.getvalue().count("same message") == 5
    logger.close()


def test_throttle_does_not_suppress_errors() -> None:
    logger = HeyRobotLogger(
        name="test", level="DEBUG", theme="dumb", timezone="UTC", throttle_sec=10.0
    )
    stream = io.StringIO()
    logger.handler.setStream(stream)
    logger._stream = stream

    for _ in range(3):
        logger.error("严重错误")
    assert stream.getvalue().count("严重错误") == 3
    logger.close()


def test_logging_spec_parsed_from_yaml_with_new_fields() -> None:
    config = DeploymentConfig.from_dict(
        {
            "logging": {
                "level": "debug",
                "theme": "light",
                "file_path": "runtime/test.log",
                "json_format": True,
                "json_file_path": "runtime/log.jsonl",
                "throttle_sec": 5.0,
            },
        }
    )
    assert config.logging.level == "DEBUG"
    assert config.logging.theme == "light"
    assert config.logging.file_path == "runtime/test.log"
    assert config.logging.json_format is True
    assert config.logging.json_file_path == "runtime/log.jsonl"
    assert config.logging.throttle_sec == 5.0
