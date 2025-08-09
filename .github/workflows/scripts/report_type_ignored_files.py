"""
prints out the name of all files containing "# type: ignore"
"""

import argparse
import os
from os import path
from os.path import join


IGNORE_LINE = "# type: ignore"


def read_file(filepath: str) -> str:
    with open(filepath) as f:
        return f.read()


def walk(target_folder: str, rel_folder: str) -> None:
    for file in os.listdir(target_folder):
        if file in [".", ".."]:
            continue
        filepath = join(target_folder, file)
        rel_path = join(rel_folder, file)
        if path.isdir(filepath):
            walk(filepath, rel_path)
        if not file.endswith(".py"):
            continue
        filecontents = read_file(filepath)
        if filecontents.strip() == "":
            continue
        file_lines = filecontents.split("\n")
        if IGNORE_LINE in [l.strip() for l in file_lines]:
            print(f"excluded from typing: {rel_path}")
            continue


def run(args: argparse.Namespace) -> None:
    walk(target_folder=args.in_dir, rel_folder=".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
