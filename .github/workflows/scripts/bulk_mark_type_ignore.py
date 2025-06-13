"""
Given path to a folder, marks all .py files in that folder tree with # type: ignore, on the top line,
followed by a blank line.
if such marking does not already exist somewhere in the file.

idempotent
"""

import argparse
import os
from os import path
from os.path import join


IGNORE_LINE = "# type: ignore"


def read_file(filepath: str) -> str:
    with open(filepath) as f:
        return f.read()


def write_file(filepath: str, contents: str) -> None:
    with open(filepath, "w") as f:
        f.write(contents)


def walk(target_folder: str) -> None:
    for file in os.listdir(target_folder):
        if file in [".", ".."]:
            continue
        filepath = join(target_folder, file)
        if path.isdir(filepath):
            walk(filepath)
        if not file.endswith(".py"):
            continue
        filecontents = read_file(filepath)
        if filecontents.strip() == "":
            continue
        file_lines = filecontents.split("\n")
        if IGNORE_LINE in [l.strip() for l in file_lines]:
            continue
        file_lines = [IGNORE_LINE, ""] + file_lines
        write_file(filepath, "\n".join(file_lines))
        print("updated", filepath)


def run(args: argparse.Namespace) -> None:
    walk(target_folder=args.in_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
