"""
Small script to emojize html files.

Inspired by: https://bitbucket.org/lbesson/bin/src/master/emojize.py
"""

import re
from pathlib import Path
from sys import argv

from emoji import emojize


def match_to_emoji(m: re.Match) -> str:
    return emojize(m.group(), language="alias")


def emojize_all(s: str) -> str:
    return re.sub(r":([0-9a-z_-]+):", match_to_emoji, s)


if __name__ == "__main__":
    dir = Path(argv[1])

    for file in dir.glob("*.html"):
        with file.open() as f:
            html = f.readlines()

        html = [emojize_all(line) for line in html]

        with file.open("w") as f:
            f.write("".join(html))
