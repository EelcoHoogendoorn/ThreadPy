

"""
abstract gevent interface
"""

import gevent


class GreenStream(gevent.Greenlet):
    """
    abstract class
    associates a greenlet with a backend-specific stream/queue object
    can be used to provide a backend neutral interface to the combination of both
    """
    def __init__(self, context, stream, body, *args, **kwargs):
        args = (self,) + args
        super(GreenStream, self).__init__(body, *args, **kwargs)
        self.context = context
        self.stream = stream


    def Yield(self):
        """canonical yield in gevent"""
        gevent.sleep()

    def _run(self):
        raise NotImplementedError()

