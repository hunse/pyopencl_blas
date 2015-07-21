import numpy as np
import pytest

import pyopencl as cl
from pyopencl.array import Array

import pyopencl_blas.blas as blas

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


tolerances = {
    'float32': dict(atol=1e-7, rtol=1e-4),
    'float64': dict(atol=1e-8, rtol=1e-5),
    'complex64': dict(atol=1e-6, rtol=1e-3),
    'complex128': dict(atol=1e-7, rtol=1e-4)
}


def to_ocl(a):
    cla = Array(queue, a.shape, a.dtype)
    cla.set(a)
    return cla


def gemv_system(queue, m, n, dtype, rng):
    A = np.zeros((m, n), dtype=dtype)
    X = np.zeros(n, dtype=dtype)
    Y = np.zeros(m, dtype=dtype)
    A[...] = rng.uniform(-1, 1, size=A.shape)
    X[...] = rng.uniform(-1, 1, size=X.shape)
    Y[...] = rng.uniform(-1, 1, size=Y.shape)
    return A, X, Y


def test_check_dtype(rng):
    dtype = 'float32'
    m, n = 3, 4
    sA, sX, sY = gemv_system(queue, m, n, dtype, rng)
    dA, dX, dY = map(lambda x: x.astype('float64'), [sA, sX, sY])
    cA, cX, cY = map(lambda x: x.astype('complex64'), [sA, sX, sY])
    zA, zX, zY = map(lambda x: x.astype('complex128'), [sA, sX, sY])

    clsA, clsX, clsY = map(to_ocl, (sA, sX, sY))
    cldA, cldX, cldY = map(to_ocl, (dA, dX, dY))
    clcA, clcX, clcY = map(to_ocl, (cA, cX, cY))
    clzA, clzX, clzY = map(to_ocl, (zA, zX, zY))

    try:
        blas.setup()

        with pytest.raises(ValueError):
            blas.gemv(queue, cldA, clsX, clsY)

        with pytest.raises(ValueError):
            blas.gemv(queue, clsA, cldX, clsY)

        with pytest.raises(ValueError):
            blas.gemv(queue, clcA, cldX, clcY)

        with pytest.raises(ValueError):
            blas.gemv(queue, clzA, cldX, clzY)
    finally:
        blas.teardown()


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('n', [7, 25, 100])
def test_blas1(n, dtype, rng):
    tols = tolerances[dtype]

    x = np.zeros(n, dtype=dtype)
    y = np.zeros(n, dtype=dtype)
    x[:] = rng.uniform(-1, 1, size=n)
    y[:] = rng.uniform(-1, 1, size=n)
    alpha = rng.uniform(0.1, 0.9)

    clx, cly = map(to_ocl, [x, y])

    try:
        blas.setup()

        blas.swap(queue, clx, cly)
        assert np.allclose(clx.get(), y, **tols)
        assert np.allclose(cly.get(), x, **tols)

        clx.set(x)
        blas.scal(queue, alpha, clx)
        assert np.allclose(clx.get(), alpha * x, **tols)

        clx.set(x)
        blas.copy(queue, clx, cly)
        assert np.allclose(cly.get(), x, **tols)

        clx.set(x)
        cly.set(y)
        blas.axpy(queue, clx, cly, alpha=alpha)
        assert np.allclose(cly.get(), alpha * x + y, **tols)

    finally:
        blas.teardown()


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('m, n', [(5, 6), (10, 10), (100, 79)])
def test_gemv(m, n, dtype, rng):
    tols = tolerances[dtype]

    A, X, Y = gemv_system(queue, m, n, dtype, rng)
    clA, clX, clY = map(to_ocl, (A, X, Y))

    try:
        blas.setup()

        # normal gemv
        clX.set(X)
        clY.fill(0)
        blas.gemv(queue, clA, clX, clY)
        assert np.allclose(clY.get(), np.dot(A, X), **tols)

        # transposed gemv
        clX.fill(0)
        clY.set(Y)
        blas.gemv(queue, clA, clY, clX, transA=True)
        assert np.allclose(clX.get(), np.dot(A.T, Y), **tols)

        # sliced gemv
        clX.set(X)
        clY.fill(0)
        blas.gemv(queue, clA[:-1, 1:], clX[:-1], clY[1:])
        Yslice = clY.get()
        assert np.allclose(Yslice[0], 0, **tols)
        assert np.allclose(Yslice[1:], np.dot(A[:-1, 1:], X[:-1]), **tols)

    finally:
        blas.teardown()


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('m, k, n', [(5, 6, 7), (10, 9, 10)])
def test_gemm(m, k, n, dtype, rng):
    tols = tolerances[dtype]

    A = np.zeros((m, k), dtype=dtype)
    B = np.zeros((k, n), dtype=dtype)
    C = np.zeros((m, n), dtype=dtype)
    CT = np.zeros((n, m), dtype=dtype)
    A[...] = rng.uniform(-1, 1, size=A.shape)
    B[...] = rng.uniform(-1, 1, size=B.shape)

    clA, clB, clC, clCT = map(to_ocl, [A, B, C, CT])

    try:
        blas.setup()

        # normal gemm
        blas.gemm(queue, clA, clB, clC)
        assert np.allclose(clC.get(), np.dot(A, B), **tols)

        # double transposed gemm
        blas.gemm(queue, clB, clA, clCT, transA=True, transB=True)
        assert np.allclose(clCT.get(), np.dot(B.T, A.T), **tols)

        # sliced gemm
        clC.fill(0)
        blas.gemm(queue, clA[:-1, 1:], clB[:-1, 1:], clC[1:, :-1])
        Cslice = clC.get()
        assert np.allclose(Cslice[0, :], 0, **tols)
        assert np.allclose(Cslice[:, -1], 0, **tols)
        assert np.allclose(Cslice[1:, :-1],
                           np.dot(A[:-1, 1:], B[:-1, 1:]), **tols)

    finally:
        blas.teardown()
