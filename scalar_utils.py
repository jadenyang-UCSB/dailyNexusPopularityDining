"""
Utilities to avoid "The truth value of an array with more than one element is ambiguous"
when using NumPy values in Python if/and/or conditions.

Use these helpers whenever you use values from NumPy or YOLO (e.g. timer["time"],
timer["direction"], or similarity scores) in boolean or comparison logic.
"""
import numpy as np


def to_int(x):
    """Convert to Python int. Safe for numpy scalars and 0-d arrays."""
    arr = np.asarray(x)
    if arr.size == 0:
        return 0
    return int(arr.flat[0])


def to_float(x):
    """Convert to Python float. Safe for numpy scalars and 0-d arrays."""
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.flat[0])


def to_bool(x):
    """Convert to Python bool. Safe for numpy scalars and 0-d arrays."""
    arr = np.asarray(x)
    if arr.size == 0:
        return False
    return bool(arr.flat[0])
