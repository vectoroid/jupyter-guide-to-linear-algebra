#! /usr/bin/env python3
import enum
import os
import pathlib
import subprocess
import typing as t

class ValidFileExtensions(enum.StrEnum):
    JUPYTERFILE = '.ipynb'
    PYTHONFILE = '.py'

def is_jupyter_file(fname: str) -> bool:
    f = pathlib.Path(fname)
    if not f.is_file():
        raise ValueError(f"Error: <{f.name}> is not a file. File conversion only works on files (of course).")

    return f.match(ValidFileExtensions.JUPYTERFILE)

def get_work_dir() -> pathlib.Path:
    return pathlib.Path(os.getcwd())

def convert_file_from_jupyter_to_marimo(fname: str, debug: bool=False) -> t.NoReturn:
    f = pathlib.Path(fname)
    if not f.is_file():
        raise ValueError(f"Error: <{f.name}> is not a file. File conversion only works on files (of course).")
    else:
        args: list = ['marimo', 'convert', f"./{f.name}", '-o', f"{f.with_suffix(ValidFileExtensions.PYTHONFILE).name}"]
        if debug:
            print(f"Preparing to convert {f.name} to {f.with_suffix(ValidFileExtensions.PYTHONFILE).name} ...")
        subprocess.run(args)

def main():
    working_dir = pathlib.Path(get_work_dir())
    jupyter_files = sorted(working_dir.glob(f"*{ValidFileExtensions.JUPYTERFILE}"))
    for f in working_dir.glob(f"*{ValidFileExtensions.JUPYTERFILE}"):
        convert_file_from_jupyter_to_marimo(f, debug=True)


if __name__ == "__main__":
    main()
