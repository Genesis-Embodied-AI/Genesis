import re

import genesis as gs


class COLORS:
    # Reference:
    # https://talyian.github.io/ansicolors/
    # https://bixense.com/clicolors/
    def __init__(self) -> None:
        pass

    @property
    def GREEN(self):
        if gs._theme == "dark":
            return "\x1b[38;5;119m"
        elif gs._theme == "light":
            return "\x1b[38;5;2m"
        elif gs._theme == "dumb":
            return ""

    @property
    def BLUE(self):
        if gs._theme == "dark":
            return "\x1b[38;5;159m"
        elif gs._theme == "light":
            return "\x1b[38;5;17m"
        elif gs._theme == "dumb":
            return ""

    @property
    def YELLOW(self):
        if gs._theme == "dark":
            return "\x1b[38;5;226m"
        elif gs._theme == "light":
            return "\x1b[38;5;3m"
        elif gs._theme == "dumb":
            return ""

    @property
    def RED(self):
        if gs._theme == "dark":
            return "\x1b[38;5;9m"
        elif gs._theme == "light":
            return "\x1b[38;5;1m"
        elif gs._theme == "dumb":
            return ""

    @property
    def CORN(self):
        if gs._theme == "dark":
            return "\x1b[38;5;11m"
        elif gs._theme == "light":
            return "\x1b[38;5;178m"
        elif gs._theme == "dumb":
            return ""

    @property
    def GRAY(self):
        if gs._theme == "dark":
            return "\x1b[38;5;247m"
        elif gs._theme == "light":
            return "\x1b[38;5;239m"
        elif gs._theme == "dumb":
            return ""

    @property
    def MINT(self):
        if gs._theme == "dark":
            return "\x1b[38;5;121m"
        elif gs._theme == "light":
            return "\x1b[38;5;23m"
        elif gs._theme == "dumb":
            return ""


class FORMATS:
    def __init__(self) -> None:
        pass

    @property
    def BOLD(self):
        if gs._theme == "dumb":
            return ""
        else:
            return "\x1b[1m"

    @property
    def ITALIC(self):
        if gs._theme == "dumb":
            return ""
        else:
            return "\x1b[3m"

    @property
    def UNDERLINE(self):
        if gs._theme == "dumb":
            return ""
        else:
            return "\x1b[4m"

    @property
    def RESET(self):
        if gs._theme == "dumb":
            return ""
        else:
            return "\x1b[0m"


def styless(text):
    pattern = re.compile(r"\x1b\[(\d+)(?:;\d+)*m")
    return pattern.sub("", text)


colors = COLORS()
formats = FORMATS()
