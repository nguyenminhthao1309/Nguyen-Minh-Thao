"""Builds the C++ core (fxquant._fxcore). Metadata lives in pyproject.toml."""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra_args = ["/O2"] if sys.platform == "win32" else ["-O3"]

ext_modules = [
    Pybind11Extension(
        "fxquant._fxcore",
        sources=[
            "cpp/bindings/module.cpp",
            "cpp/src/indicators.cpp",
            "cpp/src/labeling.cpp",
            "cpp/src/backtest.cpp",
        ],
        include_dirs=["cpp/include"],
        cxx_std=17,
        extra_compile_args=extra_args,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
