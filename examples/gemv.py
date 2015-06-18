from __future__ import print_function

import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyopencl_blas as blas

# start up OpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# start up the BLAS
blas.setup()

# generate some random data on the CPU
m, n = 5, 4
dtype = 'float32'  # also supports 'float64'

A = np.zeros((m, n), dtype=dtype)
x = np.zeros(n, dtype=dtype)
y = np.zeros(m, dtype=dtype)

rng = np.random.RandomState(1)  # change the seed to see different data
A[...] = rng.uniform(-1, 1, size=A.shape)
x[...] = rng.uniform(-1, 1, size=x.shape)
y[...] = rng.uniform(-1, 1, size=y.shape)

# allocate OpenCL memory on the device
clA = Array(queue, A.shape, A.dtype)
clx = Array(queue, x.shape, x.dtype)
cly = Array(queue, y.shape, y.dtype)

# copy data to device
clA.set(A)
clx.set(x)

# compute a matrix-vector product (gemv)
blas.gemv(queue, clA, clx, cly)

# check the result
print("Expected: ", np.dot(A, x))
print("Actual:   ", cly.get())

# try a matrix-vector product with the transpose
cly.set(y)
blas.gemv(queue, clA, cly, clx, transA=True)
print("Expected: ", np.dot(A.T, y))
print("Actual:   ", clx.get())

# tidy up the BLAS
blas.teardown()
