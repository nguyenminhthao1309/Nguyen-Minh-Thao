"""Write .vscode/c_cpp_properties.json so C++ IntelliSense finds pybind11 and Python.h.

Run by the setup scripts with the project's .venv interpreter. The paths are
specific to this computer, so the file is not committed to git.
"""

import json
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    include = ["${workspaceFolder}/cpp/include"]
    try:
        import pybind11

        include.append(Path(pybind11.get_include()).as_posix())
    except ImportError:
        pass
    python_include = sysconfig.get_paths().get("include")
    if python_include:
        include.append(Path(python_include).as_posix())
    config = {
        "configurations": [{"name": "fxquant", "includePath": include, "cppStandard": "c++17"}],
        "version": 4,
    }
    target = ROOT / ".vscode" / "c_cpp_properties.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
