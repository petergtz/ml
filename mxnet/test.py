#!/usr/local/bin/python

import mxnet as mx
import time
import mxnet.ndarray as nd
import numpy as np

start = time.time()
X  = nd.random.normal(shape=(10000,40000), ctx=mx.gpu())
# X  = nd.random.normal(shape=(10000,40000))
print time.time()-start
print X


start = time.time()
Y = np.random.randn(10000,40000)
print time.time()-start
print Y
