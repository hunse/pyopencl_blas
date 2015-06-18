from __future__ import print_function

import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyopencl_blas as blas

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# need to initialize the library
blas.setup()

dtype = 'float32'  # also supports 'float64'
x = np.array([1, 2, 3, 4], dtype=dtype)
y = np.array([4, 3, 2, 1], dtype=dtype)

clx = Array(queue, x.shape, x.dtype)
cly = Array(queue, y.shape, y.dtype)
clx.set(x)
cly.set(y)

# call a BLAS function on the arrays
blas.axpy(queue, clx, cly, alpha=0.8)
print("Expected: ", 0.8 * x + y)
print("Actual:   ", cly.get())

# clean up the library when finished
blas.teardown()
