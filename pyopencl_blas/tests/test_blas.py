import numpy as np
import pytest

import pyopencl as cl
from pyopencl.array import Array

import pyopencl_blas.blas as blas

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


def gemv_system(queue, m, n, dtype, rng):
    A = np.zeros((m, n), dtype=dtype)
    X = np.zeros(n, dtype=dtype)
    Y = np.zeros(m, dtype=dtype)
    A[...] = rng.uniform(-1, 1, size=A.shape)
    X[...] = rng.uniform(-1, 1, size=X.shape)
    Y[...] = rng.uniform(-1, 1, size=Y.shape)
    return A, X, Y


def to_ocl(a):
    cla = Array(queue, a.shape, a.dtype)
    cla.set(a)
    return cla


def test_check_dtype(rng):
    dtype = 'float32'
    m, n = 3, 4
    A, X, Y = gemv_system(queue, m, n, dtype, rng)
    dA, dX, dY = A.astype('float64'), X.astype('float64'), Y.astype('float64')

    clA, clX, clY, cldA, cldX, cldY = map(to_ocl, (A, X, Y, dA, dX, dY))

    try:
        blas.setup()

        with pytest.raises(ValueError):
            blas.gemv(queue, cldA, clX, clY)

        with pytest.raises(ValueError):
            blas.gemv(queue, clA, cldX, clY)

        # float64 not supported yet
        with pytest.raises(ValueError):
            blas.gemv(queue, cldA, cldX, cldY)
    finally:
        blas.teardown()


@pytest.mark.parametrize('m, n', [(5, 6), (10, 10), (100, 79)])
def test_gemv(m, n, rng):
    dtype = 'float32'
    args = dict(atol=1e-7, rtol=1e-4)

    A, X, Y = gemv_system(queue, m, n, dtype, rng)
    clA, clX, clY = map(to_ocl, (A, X, Y))

    try:
        blas.setup()

        # normal gemv
        clX.set(X)
        clY.fill(0)
        blas.gemv(queue, clA, clX, clY)
        assert np.allclose(clY.get(), np.dot(A, X), **args)

        # transposed gemv
        clX.fill(0)
        clY.set(Y)
        blas.gemv(queue, clA, clY, clX, transA=True)
        assert np.allclose(clX.get(), np.dot(A.T, Y), **args)

        # sliced gemv
        clX.set(X)
        clY.fill(0)
        blas.gemv(queue, clA[:-1, 1:], clX[:-1], clY[1:])
        Yslice = clY.get()
        assert np.allclose(Yslice[0], 0, **args)
        assert np.allclose(Yslice[1:], np.dot(A[:-1, 1:], X[:-1]), **args)

    finally:
        blas.teardown()
