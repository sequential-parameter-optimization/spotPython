"""
This module implements th Kriging and Surrogate classes.

"""

from .kriging import Kriging
from .surrogates import surrogates

__all__ = [
    "Kriging", "surrogates"
]
