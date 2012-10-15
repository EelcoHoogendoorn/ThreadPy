

import numpy as np

import pyopencl
import pyopencl.array
import pyopencl.tools

from ... import interface
from .ndarray import ndarray


def array_wrap(arr):
    """coerce object to be a wrapped array"""
    arr.__class__ = ndarray
    return arr


class Context(interface.Context):
    """
    what to do about stream/queue naming conflict?
    can use queue attribute internally, to avoid confusion
    but use stream property for any external accesss, for uniform interface
    """

    def __init__(self, device=0):
        print 'warning; device argument is ignored. this code is better sorted out by someone who actually has any idea whta he is doing, opencl wise'
        self.context = pyopencl.create_some_context()
        self.stream = pyopencl.CommandQueue(self.context)

##        self.memory_pool = pyopencl.tools.MemoryPool(pyopencl.tools.ImmediateAllocator())
##        self.allocator = self.memory_pool.allocate
        print 'warning; can not even get memory pooling to work in opencl...'
        self.allocator = pyopencl.CLAllocator(self.context)



    def __exit__(self, exc_type, exc_value, traceback):
        import gc
        gc.collect()



    #array allocators; only need to overload the abstract ones
    def empty(self, shape, dtype):
        return ndarray(self.stream, shape, dtype, allocator = self.allocator)
    def array(self, obj, stream = None):
        if stream is None: stream = self.stream
        """copy constructor"""
        if isinstance(obj, interface.ndarray):
            return obj.copy()
        if isinstance(obj, pyopencl.array.Array):
            arr = pyopencl.array.empty_like(obj)
            pyopencl.array.Array._copy(arr, obj)
            return array_wrap( arr)
        if isinstance(obj, np.ndarray):
            return array_wrap( pyopencl.array.to_device(stream, obj, self.allocator))

    def arange(self, *args, **kwargs):
        return pyopencl.array.arange(self.stream, *args, **kwargs)


    def stream(self, body, *args, **kwargs):
        raise




