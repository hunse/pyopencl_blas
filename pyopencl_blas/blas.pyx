include "blas_base.pyx"

from libcpp cimport bool

import numpy as np
from pyopencl.array import Array


dtype_size = {np.dtype('float32'): 4,
              np.dtype('float64'): 8}


def dtypes_str(dtypes):
    if len(dtypes) == 1:
        return "'%s'" % dtypes[0]
    else:
        return "one of %s" % dtypes


def check_dtype(args, dtypes):
    dtype = args[0].dtype
    if not all(arg.dtype == dtype for arg in args):
        raise ValueError("All arguments must have the same dtype (%s)"
                         % dtypes_str(dtypes))
    if dtype not in dtypes:
        raise ValueError("Data type must be %s" % dtypes_str(dtypes))

    return dtype


def check_array(a, ndim, name):
    if not isinstance(a, Array):
        raise ValueError("'%s' must be a PyOpenCL Array" % name)
    if not len(a.shape) == ndim:
        raise ValueError("'%s' must have %d dimensions (got %d)" %
                         (name, ndim, len(a.shape)))


def check_matrix(a, name):
    check_array(a, 2, name)


def check_vector(a, name):
    check_array(a, 1, name)


def check_shape_dim(shape, dim, target, name):
    if shape[dim] != target:
        raise ValueError("'%s.shape[%d]' must be %d (got %d)" %
                         (name, dim, target, shape[dim]))


cdef extern from "clBLAS.h":
    clblasStatus clblasSetup()
    void clblasTeardown()


def setup():
    cdef clblasStatus err = clblasSetup()
    if err != clblasSuccess:
        raise RuntimeError("Failed to setup clBLAS (Error %d)" % err)


def teardown():
    clblasTeardown()


########## SWAP ##########
cdef extern from "clBLAS.h":
    clblasStatus clblasSswap(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    clblasStatus clblasDswap(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)


def swap(queue, x, y):
    """y, x = x, y"""
    dtype = check_dtype([x, y], ['float32', 'float64'])
    check_vector(x, 'x')
    check_vector(y, 'y')

    cdef size_t N = x.shape[0]
    check_shape_dim(y.shape, 0, N, 'y')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem xdata = <cl_mem><size_t>x.base_data.int_ptr
    cdef size_t offx = x.offset / element_size
    cdef int incx = x.strides[0] / element_size
    cdef cl_mem ydata = <cl_mem><size_t>y.base_data.int_ptr
    cdef size_t offy = y.offset / element_size
    cdef int incy = y.strides[0] / element_size

    cdef cl_uint numCommandQueues = 1
    cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_uint numEventsInWaitList = 0
    cdef cl_event *eventWaitList = NULL
    cdef cl_event event = NULL

    cdef clblasStatus
    if dtype == np.dtype('float32'):
        err = clblasSswap(
            N, xdata, offx, incx, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    elif dtype == np.dtype('float64'):
        err = clblasDswap(
            N, xdata, offx, incx, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'swap' failed: %s" % get_status_message(err))


########## SCAL ##########
cdef extern from "clBLAS.h":
    clblasStatus clblasSscal(
        size_t N,
        cl_float alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    clblasStatus clblasDscal(
        size_t N,
        cl_double alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)


def scal(queue, alpha, x):
    """x <- alpha * x"""
    dtype = check_dtype([x], ['float32', 'float64'])
    check_vector(x, 'x')

    cdef size_t N = x.shape[0]

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem xdata = <cl_mem><size_t>x.base_data.int_ptr
    cdef size_t offx = x.offset / element_size
    cdef int incx = x.strides[0] / element_size

    cdef cl_uint numCommandQueues = 1
    cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_uint numEventsInWaitList = 0
    cdef cl_event *eventWaitList = NULL
    cdef cl_event event = NULL

    cdef clblasStatus
    if dtype == np.dtype('float32'):
        err = clblasSscal(
            N, <cl_float>alpha, xdata, offx, incx,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    elif dtype == np.dtype('float64'):
        err = clblasDscal(
            N, <cl_double>alpha, xdata, offx, incx,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'scal' failed: %s" % get_status_message(err))


########## COPY ##########
cdef extern from "clBLAS.h":
    clblasStatus clblasScopy(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    clblasStatus clblasDcopy(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)


def copy(queue, x, y):
    """y, x = x, y"""
    dtype = check_dtype([x, y], ['float32', 'float64'])
    check_vector(x, 'x')
    check_vector(y, 'y')

    cdef size_t N = x.shape[0]
    check_shape_dim(y.shape, 0, N, 'y')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem xdata = <cl_mem><size_t>x.base_data.int_ptr
    cdef size_t offx = x.offset / element_size
    cdef int incx = x.strides[0] / element_size
    cdef cl_mem ydata = <cl_mem><size_t>y.base_data.int_ptr
    cdef size_t offy = y.offset / element_size
    cdef int incy = y.strides[0] / element_size

    cdef cl_uint numCommandQueues = 1
    cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_uint numEventsInWaitList = 0
    cdef cl_event *eventWaitList = NULL
    cdef cl_event event = NULL

    cdef clblasStatus
    if dtype == np.dtype('float32'):
        err = clblasScopy(
            N, xdata, offx, incx, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    elif dtype == np.dtype('float64'):
        err = clblasDcopy(
            N, xdata, offx, incx, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'copy' failed: %s" % get_status_message(err))


########## AXPY ##########
cdef extern from "clBLAS.h":
    clblasStatus clblasSaxpy(
        size_t N,
        cl_float alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    clblasStatus clblasDaxpy(
        size_t N,
        cl_double alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)


def axpy(queue, x, y, alpha=1.0):
    """y <- alpha * x + y"""
    dtype = check_dtype([x, y], ['float32', 'float64'])
    check_vector(x, 'x')
    check_vector(y, 'y')

    cdef size_t N = x.shape[0]
    check_shape_dim(y.shape, 0, N, 'y')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem xdata = <cl_mem><size_t> x.base_data.int_ptr
    cdef size_t offx = x.offset / element_size
    cdef int incx = x.strides[0] / element_size
    cdef cl_mem ydata = <cl_mem><size_t>y.base_data.int_ptr
    cdef size_t offy = y.offset / element_size
    cdef int incy = y.strides[0] / element_size

    cdef cl_uint numCommandQueues = 1
    cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_uint numEventsInWaitList = 0
    cdef cl_event *eventWaitList = NULL
    cdef cl_event event = NULL

    cdef clblasStatus
    if dtype == np.dtype('float32'):
        err = clblasSaxpy(
            N, <cl_float>alpha, xdata, offx, incx, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    elif dtype == np.dtype('float64'):
        err = clblasDaxpy(
            N, <cl_double>alpha, xdata, offx, incx, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'axpy' failed: %s" % get_status_message(err))


# TODO: implement scratch buffers for the following functions

# ########## DOT ##########
# cdef extern from "clBLAS.h":
#     clblasStatus clblasSdot(
#         size_t N,
#         cl_float alpha,
#         cl_mem X,
#         size_t offx,
#         int incx,
#         cl_mem Y,
#         size_t offy,
#         int incy,
#         cl_uint numCommandQueues,
#         cl_command_queue *commandQueues,
#         cl_uint numEventsInWaitList,
#         const cl_event *eventWaitList,
#         cl_event *events)
#     clblasStatus clblasDdot(
#         size_t N,
#         cl_double alpha,
#         cl_mem X,
#         size_t offx,
#         int incx,
#         cl_mem Y,
#         size_t offy,
#         int incy,
#         cl_uint numCommandQueues,
#         cl_command_queue *commandQueues,
#         cl_uint numEventsInWaitList,
#         const cl_event *eventWaitList,
#         cl_event *events)


# def dot(queue, x, y, d):
#     """d <- dot(x, y)"""
#     dtype = check_dtype([x, y, d], ['float32', 'float64'])
#     check_vector(x, 'x')
#     check_vector(y, 'y')
#     check_scalar(d, 'd')

#     cdef size_t N = x.shape[0]
#     check_shape_dim(y.shape, 0, N, 'y')

#     cdef size_t element_size = dtype_size[dtype]
#     cdef cl_mem ddata = <cl_mem><size_t>d.base_data.int_ptr
#     cdef size_t offd = d.offset / element_size
#     cdef cl_mem xdata = <cl_mem><size_t>x.base_data.int_ptr
#     cdef size_t offx = x.offset / element_size
#     cdef int incx = x.strides[0] / element_size
#     cdef cl_mem ydata = <cl_mem><size_t>y.base_data.int_ptr
#     cdef size_t offy = y.offset / element_size
#     cdef int incy = y.strides[0] / element_size

#     cdef cl_uint numCommandQueues = 1
#     cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
#     cdef cl_uint numEventsInWaitList = 0
#     cdef cl_event *eventWaitList = NULL
#     cdef cl_event event = NULL

#     cdef clblasStatus
#     if dtype == np.dtype('float32'):
#         err = clblasSdot(
#             N, ddata, offd, xdata, offx, incx, ydata, offy, incy,
#             numCommandQueues, &commandQueue,
#             numEventsInWaitList, eventWaitList, &event)
#     elif dtype == np.dtype('float64'):
#         err = clblasDdot(
#             N, ddata, offd, xdata, offx, incx, ydata, offy, incy,
#             numCommandQueues, &commandQueue,
#             numEventsInWaitList, eventWaitList, &event)
#     else:
#         raise ValueError("Unrecognized dtype '%s'" % dtype)

#     if err != clblasSuccess:
#         raise RuntimeError("'dot' failed: %s" % get_status_message(err))


# ########## NRM2 ##########
# cdef extern from "clBLAS.h":
#     clblasStatus clblasSnrm2(
#         size_t N,
#         cl_mem NRM2,
#         size_t offNRM2,
#         cl_mem X,
#         size_t offx,
#         int incx,
#         cl_uint numCommandQueues,
#         cl_command_queue *commandQueues,
#         cl_uint numEventsInWaitList,
#         const cl_event *eventWaitList,
#         cl_event *events)
#     clblasStatus clblasDnrm2(
#         size_t N,
#         cl_mem NRM2,
#         size_t offNRM2,
#         cl_mem X,
#         size_t offx,
#         int incx,
#         cl_uint numCommandQueues,
#         cl_command_queue *commandQueues,
#         cl_uint numEventsInWaitList,
#         const cl_event *eventWaitList,
#         cl_event *events)


# def nrm2(queue, x, d):
#     """d <- sqrt(dot(x', x))"""
#     dtype = check_dtype([x, d], ['float32', 'float64'])
#     check_vector(x, 'x')
#     check_scalar(d, 'd')

#     cdef size_t N = x.shape[0]

#     cdef size_t element_size = dtype_size[dtype]
#     cdef cl_mem ddata = <cl_mem><size_t>d.base_data.int_ptr
#     cdef size_t offd = d.offset / element_size
#     cdef cl_mem xdata = <cl_mem><size_t>x.base_data.int_ptr
#     cdef size_t offx = x.offset / element_size
#     cdef int incx = x.strides[0] / element_size

#     cdef cl_uint numCommandQueues = 1
#     cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
#     cdef cl_uint numEventsInWaitList = 0
#     cdef cl_event *eventWaitList = NULL
#     cdef cl_event event = NULL

#     cdef clblasStatus
#     if dtype == np.dtype('float32'):
#         err = clblasSnrm2(
#             N, ddata, offd, xdata, offx, incx,
#             numCommandQueues, &commandQueue,
#             numEventsInWaitList, eventWaitList, &event)
#     elif dtype == np.dtype('float64'):
#         err = clblasDnrm2(
#             N, ddata, offd, xdata, offx, incx,
#             numCommandQueues, &commandQueue,
#             numEventsInWaitList, eventWaitList, &event)
#     else:
#         raise ValueError("Unrecognized dtype '%s'" % dtype)

#     if err != clblasSuccess:
#         raise RuntimeError("'nrm2' failed: %s" % get_status_message(err))


# ########## ASUM ##########
# cdef extern from "clBLAS.h":
#     clblasStatus clblasSasum(
#         size_t N,
#         cl_mem ASUM,
#         size_t offASUM,
#         cl_mem X,
#         size_t offx,
#         int incx,
#         cl_uint numCommandQueues,
#         cl_command_queue *commandQueues,
#         cl_uint numEventsInWaitList,
#         const cl_event *eventWaitList,
#         cl_event *events)
#     clblasStatus clblasDasum(
#         size_t N,
#         cl_mem ASUM,
#         size_t offASUM,
#         cl_mem X,
#         size_t offx,
#         int incx,
#         cl_uint numCommandQueues,
#         cl_command_queue *commandQueues,
#         cl_uint numEventsInWaitList,
#         const cl_event *eventWaitList,
#         cl_event *events)


# def asum(queue, x, d):
#     """d <- sum(abs(x))"""
#     dtype = check_dtype([x, d], ['float32', 'float64'])
#     check_vector(x, 'x')
#     check_scalar(d, 'd')

#     cdef size_t N = x.shape[0]

#     cdef size_t element_size = dtype_size[dtype]
#     cdef cl_mem ddata = <cl_mem><size_t>d.base_data.int_ptr
#     cdef size_t offd = d.offset / element_size
#     cdef cl_mem xdata = <cl_mem><size_t>x.base_data.int_ptr
#     cdef size_t offx = x.offset / element_size
#     cdef int incx = x.strides[0] / element_size

#     cdef cl_uint numCommandQueues = 1
#     cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
#     cdef cl_uint numEventsInWaitList = 0
#     cdef cl_event *eventWaitList = NULL
#     cdef cl_event event = NULL

#     cdef clblasStatus
#     if dtype == np.dtype('float32'):
#         err = clblasSasum(
#             N, ddata, offd, xdata, offx, incx,
#             numCommandQueues, &commandQueue,
#             numEventsInWaitList, eventWaitList, &event)
#     elif dtype == np.dtype('float64'):
#         err = clblasDasum(
#             N, ddata, offd, xdata, offx, incx,
#             numCommandQueues, &commandQueue,
#             numEventsInWaitList, eventWaitList, &event)
#     else:
#         raise ValueError("Unrecognized dtype '%s'" % dtype)

#     if err != clblasSuccess:
#         raise RuntimeError("'asum' failed: %s" % get_status_message(err))


########## GEMV ##########
cdef extern from "clBLAS.h":
    clblasStatus clblasSgemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        cl_float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem x,
        size_t offx,
        int incx,
        cl_float beta,
        cl_mem y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    clblasStatus clblasDgemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        cl_double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem x,
        size_t offx,
        int incx,
        cl_double beta,
        cl_mem y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)


def gemv(queue, A, x, y, bool transA=False, alpha=1.0, beta=0.0):
    """y <- alpha * dot(A, x) + beta * y"""
    dtype = check_dtype([A, x, y], ['float32', 'float64'])
    check_matrix(A, 'A')
    check_vector(x, 'x')
    check_vector(y, 'y')

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]
    check_shape_dim(x.shape, 0, (M if transA else N), 'x')
    check_shape_dim(y.shape, 0, (N if transA else M), 'y')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem Adata = <cl_mem><size_t>A.base_data.int_ptr
    cdef size_t offA = A.offset / element_size
    cdef size_t lda = A.strides[0] / element_size
    cdef cl_mem xdata = <cl_mem><size_t>x.base_data.int_ptr
    cdef size_t offx = x.offset / element_size
    cdef int incx = x.strides[0] / element_size
    cdef cl_mem ydata = <cl_mem><size_t>y.base_data.int_ptr
    cdef size_t offy = y.offset / element_size
    cdef int incy = y.strides[0] / element_size

    cdef cl_uint numCommandQueues = 1
    cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_uint numEventsInWaitList = 0
    cdef cl_event *eventWaitList = NULL
    cdef cl_event event = NULL

    cdef clblasStatus
    if dtype == np.dtype('float32'):
        err = clblasSgemv(
            clblasRowMajor, clblasTrans if transA else clblasNoTrans,
            M, N, <cl_float>alpha, Adata, offA, lda,
            xdata, offx, incx, <cl_float>beta, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    elif dtype == np.dtype('float64'):
        err = clblasDgemv(
            clblasRowMajor, clblasTrans if transA else clblasNoTrans,
            M, N, <cl_double>alpha, Adata, offA, lda,
            xdata, offx, incx, <cl_double>beta, ydata, offy, incy,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'gemv' failed: %s" % get_status_message(err))


cdef extern from "clBLAS.h":
    clblasStatus clblasSgemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        cl_float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        cl_float beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    clblasStatus clblasDgemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        cl_double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        cl_double beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)


def gemm(queue, A, B, C, transA=False, transB=False,
         float alpha=1.0, float beta=0.0):
    """C <- alpha * dot(A, B) + beta * C"""
    dtype = check_dtype([A, B, C], ['float32', 'float64'])
    check_matrix(A, 'A')
    check_matrix(B, 'B')
    check_matrix(C, 'C')

    cdef size_t M = A.shape[1 if transA else 0]
    cdef size_t K = A.shape[0 if transA else 1]
    cdef size_t N = B.shape[0 if transB else 1]
    check_shape_dim(B.shape, 1 if transB else 0, K, 'B')
    check_shape_dim(C.shape, 0, M, 'C')
    check_shape_dim(C.shape, 1, N, 'C')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem Adata = <cl_mem><size_t>A.base_data.int_ptr
    cdef size_t offA = A.offset / element_size
    cdef size_t lda = A.strides[0] / element_size
    cdef cl_mem Bdata = <cl_mem><size_t>B.base_data.int_ptr
    cdef size_t offB = B.offset / element_size
    cdef size_t ldb = B.strides[0] / element_size
    cdef cl_mem Cdata = <cl_mem><size_t>C.base_data.int_ptr
    cdef size_t offC = C.offset / element_size
    cdef size_t ldc = C.strides[0] / element_size

    cdef cl_uint numCommandQueues = 1
    cdef cl_command_queue commandQueue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_uint numEventsInWaitList = 0
    cdef cl_event *eventWaitList = NULL
    cdef cl_event event = NULL

    if dtype == np.dtype('float32'):
        err = clblasSgemm(
            clblasRowMajor,
            clblasTrans if transA else clblasNoTrans,
            clblasTrans if transB else clblasNoTrans,
            M, N, K, <cl_float>alpha, Adata, offA, lda, Bdata, offB, ldb,
            <cl_float>beta, Cdata, offC, ldc,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    elif dtype == np.dtype('float64'):
        err = clblasDgemm(
            clblasRowMajor,
            clblasTrans if transA else clblasNoTrans,
            clblasTrans if transB else clblasNoTrans,
            M, N, K, <cl_double>alpha, Adata, offA, lda, Bdata, offB, ldb,
            <cl_double>beta, Cdata, offC, ldc,
            numCommandQueues, &commandQueue,
            numEventsInWaitList, eventWaitList, &event)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)


    if err != clblasSuccess:
        raise RuntimeError("'gemm' failed: %s" % get_status_message(err))
