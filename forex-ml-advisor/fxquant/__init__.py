"""fxquant: machine-learning forex market analysis and trade advice (Python + C++)."""

from .config import Config
from .core import BACKEND

__all__ = ["Config", "BACKEND", "ForexAdvisor"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "ForexAdvisor":  # lazy: keeps `import fxquant` light
        from .advisor import ForexAdvisor

        return ForexAdvisor
    raise AttributeError(name)
