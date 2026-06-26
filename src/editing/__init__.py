"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from .base import EditMethod, EditOutcome
from .registry import EDIT_METHODS, get_edit_method

__all__ = ["EditMethod", "EditOutcome", "EDIT_METHODS", "get_edit_method"]
