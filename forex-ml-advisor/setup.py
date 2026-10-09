"""Builds the optional C++ core (fxquant._fxcore). Metadata lives in pyproject.toml.

The extension is optional: on a machine without a C++ compiler the install
still succeeds and fxquant uses its NumPy implementation (same results, slower).
"""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra_args = ["/O2"] if sys.platform == "win32" else ["-O3"]

COMPILER_HELP = {
    "win32": "install 'Build Tools for Visual Studio' (C++ workload): "
             "https://visualstudio.microsoft.com/visual-cpp-build-tools/",
    "darwin": "install the Xcode command line tools: xcode-select --install",
}


class OptionalBuildExt(build_ext):
    """Never fail the install because the C++ core could not be compiled."""

    def run(self):
        try:
            super().run()
        except Exception as exc:  # no compiler, missing headers, ...
            self._explain(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._explain(exc)

    @staticmethod
    def _explain(exc):
        hint = COMPILER_HELP.get(sys.platform, "install a C++17 compiler (e.g. sudo apt install g++)")
        print("\n" + "=" * 78)
        print("NOTE: the optional C++ core was not built, so fxquant will use its NumPy")
        print("implementation (identical results, slower). To enable C++, " + hint)
        print("and run `pip install -e .` again.")
        print(f"Reason: {exc}")
        print("=" * 78 + "\n")


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
        optional=True,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
