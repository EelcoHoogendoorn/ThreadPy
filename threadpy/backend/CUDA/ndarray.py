
"""
CUDA ndarray wrapper object
"""

from .. import interface
from pycuda import gpuarray

class ndarray(gpuarray.GPUArray, interface.ndarray):
    def copy(self, stream = None):
        """
        default pycuda functions do not include a device-device copy method, for some reason
        """
        if not self.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                    "be used as arguments to this operation")

        result = self._new_like_me()

        func = gpuarray.elementwise.get_copy_kernel(self.dtype, self.dtype)
        func.prepared_async_call(self._grid, self._block, stream,
                result.gpudata, self.gpudata,
                self.mem_size)

        return result