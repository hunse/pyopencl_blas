include "blas_base.pyx"

from libcpp cimport bool

from pyopencl.array import Array


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


cdef extern from "clBLAS.h":
    clblasStatus clblasSetup()
    void clblasTeardown()


def setup():
    cdef clblasStatus err = clblasSetup()
    if err != clblasSuccess:
        raise RuntimeError("Failed to setup clBLAS (Error %d)" % err)


def teardown():
    clblasTeardown()


cdef extern from "clBLAS.h":
    clblasStatus clblasSgemv(
        clblasOrder order, clblasTranspose transA, size_t M, size_t N,
        cl_float alpha, const cl_mem A, size_t offA, size_t lda,
        const cl_mem x, size_t offx, int incx,
        cl_float beta, cl_mem y, size_t offy, int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)


def gemv(queue, A, X, Y, bool transA=False, float alpha=1.0, float beta=0.0):
    """Y <- beta * Y + alpha * dot(AT, X)
    where AT = A.T if transA else A
    """
    check_dtype([A, X, Y], ['float32'])
    check_matrix(A, 'A')
    check_vector(X, 'X')
    check_vector(Y, 'Y')

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]
    assert X.shape[0] == (M if transA else N)
    assert Y.shape[0] == (N if transA else M)

    cdef size_t float_size = 4
    cdef cl_mem Adata = <cl_mem><int>A.base_data.int_ptr
    cdef size_t offA = A.offset / float_size
    cdef size_t lda = A.strides[0] / float_size
    cdef cl_mem Xdata = <cl_mem><int>X.base_data.int_ptr
    cdef size_t offx = X.offset / float_size
    cdef int incx = X.strides[0] / float_size
    cdef cl_mem Ydata = <cl_mem><int>Y.base_data.int_ptr
    cdef size_t offy = Y.offset / float_size
    cdef int incy = Y.strides[0] / float_size

    cdef cl_uint numCommandQueues = 1
    cdef cl_command_queue commandQueue = <cl_command_queue><int>queue.int_ptr
    cdef cl_uint numEventsInWaitList = 0
    cdef cl_event *eventWaitList = NULL
    cdef cl_event event = NULL

    cdef clblasStatus err = clblasSgemv(
        clblasRowMajor, clblasTrans if transA else clblasNoTrans,
        M, N, alpha, Adata, offA, lda,
        Xdata, offx, incx, beta, Ydata, offy, incy,
        numCommandQueues, &commandQueue,
        numEventsInWaitList, eventWaitList, &event)

    if err != clblasSuccess:
        raise RuntimeError("'gemv' failed: %s" % get_status_message(err))
