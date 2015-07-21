pyopencl_blas
=============

PyOpenCL wrappers for AMD clMathLibraries.clBLAS.

Currently only supports a small subset of the BLAS library
(specifically the SWAP, SCAL, COPY, AXPY, GEMV, and GEMM
families of functions),
and only for real numbers (i.e., 32-bit or 64-bit floats).


Installation
------------

First, install [PyOpenCL](https://pypi.python.org/pypi/pyopencl)
as per the [instructions](http://wiki.tiker.net/PyOpenCL/Installation).
You will also need to install [Cython](http://cython.org/).

Download the
[latest clBLAS release](https://github.com/clMathLibraries/clBLAS/releases)
(2.4.0 at the time of writing),
and unpack it somewhere (I suggest unpacking to `/opt/`).
Inside, you will find an `include` directory and a `lib` directory
(called `lib64` on 64-bit machines).
Your machine will need to know where to find the libraries
when running the program.
On Linux, you can do this by putting a file in `/etc/ld.so.conf.d/`
that contains the full path to the `lib` directory (the file name must end in `.conf`).
Then call `sudo ldconfig` (you can do `sudo ldconfig -v | grep libclBLAS`
to make sure that the library has been detected).
I am not sure how to add the libraries on other systems.

Open up ``setup.py`` and change the include dirs in the extension
to target your OpenCL include directory
and your clBLAS include directory (which you just installed), respectively.
Also change the library dir to target
your clBLAS library directory (again, which you just installed).

Then, build the project:

    python setup.py build_ext --inplace

It should compile without errors, and create `pyopencl_blas/blas.so`
(as well as a corresponding `blas.cpp` file).

You can now install the project:

    python setup.py install --user

or do a "developer" install:

    python setup.py develop --user

The latter will mean that changes to this source directory will show up
when you import the package (making it easy to develop).
You do not need the `--user` flag if installing to a virtualenv
(they're great; check them out!).


Usage
-----
The basic usage is to start up PyOpenCL as usual,
create some PyOpenCL Arrays,
and pass them to the BLAS functions.

```python
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
```

See the `examples` folder for more examples.
