"""Tiny helper: build a .ipynb from a list of cells, then execute it in place.

Usage from a build script:
    from nb import md, code, build
    build("path/to/analysis.ipynb", [md("# Title"), code("import pandas as pd")])
"""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def md(src):
    return nbformat.v4.new_markdown_cell(src.strip())


def code(src):
    return nbformat.v4.new_code_cell(src.strip())


def build(path, cells, timeout=600):
    path = Path(path)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)
    print(f"executed -> {path}")


if __name__ == "__main__":
    sys.exit("import this module from a build script")
