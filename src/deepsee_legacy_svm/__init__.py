from .version import __version__
from .io import load_table
from .binary import train_binary_legacy
from .multiclass import train_multiclass_legacy

__all__ = [
    "__version__",
    "load_table",
    "train_binary_legacy",
    "train_multiclass_legacy",
]
