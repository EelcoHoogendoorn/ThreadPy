
"""
cuda context wrapper object
"""
import numpy as np

from .. import interface
from .GreenStream import GreenStream
from .ndarray import ndarray

from pycuda import driver as drv
from pycuda import gpuarray
import pycuda.tools

def array_wrap(arr):
    """coerce object to be a wrapped array"""
    arr.__class__ = ndarray
    return arr


import scikits.cuda.fft as cu_fft
class FFT(object):
    """
    numpy work-alike fft namespace, wrapping cufft
    abstracts away plan creation
    """
    def __init__(self, context):
        self.context = context
        self.plan_cache = {}
        self.rtype = np.float32
        self.ctype = np.complex64

    def flush(self):
        """flush plan cache; do we ever need this though? easier to recreate object i figure"""
        self.plan_cache = {}


    #mimick numpy fft interface on gpu
    def get_plan(self, cache, *args):
        if not args in self.plan_cache:
            plan = cu_fft.Plan(*args)
            if cache:
                self.plan_cache[args] = plan
        else:
            plan = self.plan_cache[args]
        return plan

    #public interface; only partially supported as of now
    def rfft2(self, i, o = None, cache = True):
        shape = i.shape[:-2]
        rshape = i.shape[-2:]
        cshape = (rshape[0], rshape[1]/2+1)
        batch = np.prod(shape, dtype=np.int)
        plan = self.get_plan(cache, rshape, self.rtype, self.ctype, batch)
        if o is None:
            o = self.context.empty(shape+cshape, self.ctype)
        cu_fft.fft(i, o, plan, scale=False)
        return o
    def irfft2(self, i, o = None, cache = True):
        shape = i.shape[:-2]
        cshape = i.shape[-2:]
        rshape = (cshape[0], (cshape[1]-1)*2)
        batch = np.prod(shape, dtype=np.int)
        plan = self.get_plan(cache, rshape, self.ctype, self.rtype, batch)
        if o is None:
            o = self.context.empty(shape+rshape, self.rtype)
        cu_fft.ifft(i, o, plan, scale=True)
        return o


class Context(interface.Context):
    """
    mangle cuda into a clean interface
    """
    def __init__(self, device=0):
        drv.init()

        self.device = drv.Device(device)
        self.context = self.device.make_context()

        self.memory_pool = pycuda.tools.DeviceMemoryPool()

        #init fft object
        self.fft = FFT(self)


    def __exit__(self, exc_type, exc_value, traceback):
        import gc
        gc.collect()
        self.context.pop()



    #array allocators; only need to overload the abstract ones
    def empty(self, shape, dtype):
        return ndarray(shape, dtype, allocator = self.memory_pool.allocate)
    def array(self, obj, stream = None):
        """copy constructor"""
        if isinstance(obj, interface.ndarray):
            return obj.copy()
        if isinstance(obj, gpuarray.GPUArray):
            return
        if isinstance(obj, np.ndarray):
            return array_wrap( gpuarray.to_gpu(obj, self.mempool.allocate))

    def arange(self, *args, **kwargs):
        return gpuarray.arange(*args, **kwargs)


    def stream(self, body, *args, **kwargs):
        """
        branch a stream off the main context object
        stream objects themselves should support this too
        model is that a stream always has a handle to its context
        so this should be the only public allocator, really
        """
        return GreenStream(self, body, *args, **kwargs)