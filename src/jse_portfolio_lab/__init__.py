"""
JSE Portfolio Optimiser Lab
============================

A portfolio-grade quantitative research project analyzing portfolio construction
methods on JSE Top-40 equities.

Author: Tjay Burger
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Tjay Burger"

from . import data
from . import optimization
from . import backtesting
from . import metrics
from . import utils

__all__ = ["data", "optimization", "backtesting", "metrics", "utils"]
