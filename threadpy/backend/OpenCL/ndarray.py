
"""
CUDA ndarray wrapper object
"""

from ... import interface
from pyopencl import array

class ndarray(array.Array, interface.ndarray):
    pass