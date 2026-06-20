"""Hey Robot 日志样式系统。"""

import re


class Colors:
    """带主题的 ANSI 颜色码。"""

    def __init__(self, theme: str = "dark"):
        self._theme = theme

    @property
    def green(self):
        if self._theme == "dark":
            return "\x1b[38;5;119m"
        if self._theme == "light":
            return "\x1b[38;5;2m"
        if self._theme == "dumb":
            return ""
        return ""

    @property
    def blue(self):
        if self._theme == "dark":
            return "\x1b[38;5;159m"
        if self._theme == "light":
            return "\x1b[38;5;17m"
        if self._theme == "dumb":
            return ""
        return ""

    @property
    def yellow(self):
        if self._theme == "dark":
            return "\x1b[38;5;226m"
        if self._theme == "light":
            return "\x1b[38;5;3m"
        if self._theme == "dumb":
            return ""
        return ""

    @property
    def red(self):
        if self._theme == "dark":
            return "\x1b[38;5;9m"
        if self._theme == "light":
            return "\x1b[38;5;1m"
        if self._theme == "dumb":
            return ""
        return ""

    @property
    def mint(self):
        if self._theme == "dark":
            return "\x1b[38;5;121m"
        if self._theme == "light":
            return "\x1b[38;5;23m"
        if self._theme == "dumb":
            return ""
        return ""

    @property
    def gray(self):
        if self._theme == "dark":
            return "\x1b[38;5;247m"
        if self._theme == "light":
            return "\x1b[38;5;239m"
        if self._theme == "dumb":
            return ""
        return ""


class Formats:
    """带主题的 ANSI 格式码。"""

    def __init__(self, theme: str = "dark"):
        self._theme = theme

    @property
    def bold(self):
        if self._theme == "dumb":
            return ""
        return "\x1b[1m"

    @property
    def italic(self):
        if self._theme == "dumb":
            return ""
        return "\x1b[3m"

    @property
    def underline(self):
        if self._theme == "dumb":
            return ""
        return "\x1b[4m"

    @property
    def reset(self):
        if self._theme == "dumb":
            return ""
        return "\x1b[0m"


def styless(text: str) -> str:
    """移除文本中的 ANSI 颜色码。"""
    pattern = re.compile(r"\x1b\[(\d+)(?:;\d+)*m")
    return pattern.sub("", text)


COLORS = Colors
FORMATS = Formats
