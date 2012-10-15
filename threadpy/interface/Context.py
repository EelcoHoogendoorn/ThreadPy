
"""
abstract context interface
"""


import numpy as np




class Context(object):
    """
    uniform interface, regardless of backend
    aims to be a numpy work-alike
    """

    #context management
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        raise
    def __hash__(self):
        return id(self)



    #complete set of standard allocators
    def array(self, obj):
        raise
    def empty(self, shape, dtype):
        raise
    def empty_like(self, arr):
        return self.empty(arr.shape, arr.dtype)
    def zeros(self, shape, dtype):
        return self.filled(shape, dtype, 0)
    def zeros_like(self, arr):
        return self.zeros(arr.shape, arr.dtype)
    def ones(self, shape, dtype):
        return self.filled(shape, dtype, 1)
    def ones_like(self, arr):
        return self.ones(arr.shape, arr.dtype)
    def filled(self, shape, dtype, value):
        arr = self.empty(shape, dtype)
        arr.fill(value)
        return arr
    def filled_like(self, arr, value):
        return self.filled(arr.shape, arr.dtype, value)

    #numpy defaults; may be overridden with backend-specifics
    def arange(self, *args, **kwargs):
        return self.array(np.arange(*args, **kwargs))

    #host/device transfer
    def upload(self, array):
        raise
    def download(self, gpuarray):
        raise

    #algorithms
    def fft(self):
        """provide an fft namespace similar to the one provided in numpy"""
        raise

    def sorting(self):
        """
        provide a rich set of sorting functionality.
        arguably, numpy is lacking in this regard
        but gpus even moreso
        """
        raise



