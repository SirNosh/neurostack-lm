"""Dense NeuroStack Experiment 2.

Experiment 1 remains frozen under ``src.stage1r`` and its published tags.
"""

from .data import Experiment2Example
from .dense_adapters import DenseAdapterBank

__all__ = ["DenseAdapterBank", "Experiment2Example"]
