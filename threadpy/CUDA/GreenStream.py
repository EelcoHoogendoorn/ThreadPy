

"""
"""

from .. import interface

from pycuda import driver as drv

class GreenStream(interface.GreenStream):
    """
    abstract class
    associates a greenlet with a backend-specific stream/queue object
    can be used to provide a backend neutral interface to the combination of both

    also, we may place utility function here, which are just copies of the context/array ones
    which auto-bind to this stream
    """
    def __init__(self, context, body, *args, **kwargs):
        stream = drv.Stream()
        super(GreenStream, self).__init__(context, stream, body, *args, **kwargs)

