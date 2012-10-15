
import numpy as np
from .CUDA import Context
##import gevent

def body(self, i):
    q = self.context.ones((3,3), np.float32) * i
    self.Yield()
    z = self.context.array(q)
    print z.__class__
    r = self.context.fft.rfft2(q)
    print r.__class__
    print r



with Context(device=0) as ctx:

    #do some work in the default stream
    q = ctx.arange(12, dtype=np.float32).reshape((3,4))
    print q
    z = ctx.fft.rfft2(q)
##    z=q
    print z
    print z.__class__
    print ctx.fft.irfft2(z)
##    gevent.sleep()
quit()

##    #spawn some greenstreams
##    #can we abstract away the pool as well?
##    pool = [ctx.stream(body, i) for i in xrange(10)]
##    for s in pool: s.start()
##
##    #main loop
##    while not all(s.ready() for s in pool):
##        gevent.sleep()
##

