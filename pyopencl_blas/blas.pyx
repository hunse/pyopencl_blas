import cython
from libcpp cimport bool

from pyopencl.array import Array


cdef extern from "clBLAS.h":
    ctypedef enum clblasStatus:
        clblasSuccess
        clblasInvalidValue
        clblasInvalidCommandQueue
        clblasInvalidContext
        clblasInvalidMemObject
        clblasInvalidDevice
        clblasInvalidEventWaitList
        clblasOutOfResources
        clblasOutOfHostMemory
        clblasInvalidOperation
        clblasCompilerNotAvailable
        clblasBuildProgramFailure
        clblasNotImplemented
        clblasNotInitialized
        clblasInvalidMatA
        clblasInvalidMatB
        clblasInvalidMatC
        clblasInvalidVecX
        clblasInvalidVecY
        clblasInvalidDim
        clblasInvalidLeadDimA
        clblasInvalidLeadDimB
        clblasInvalidLeadDimC
        clblasInvalidIncX
        clblasInvalidIncY
        clblasInsufficientMemMatA
        clblasInsufficientMemMatB
        clblasInsufficientMemMatC
        clblasInsufficientMemVecX
        clblasInsufficientMemVecY

    ctypedef float cl_float
    ctypedef unsigned int cl_uint

    struct _cl_mem:
        pass
    struct _cl_command_queue:
        pass
    struct _cl_event:
        pass

    ctypedef _cl_mem* cl_mem
    ctypedef _cl_command_queue* cl_command_queue
    ctypedef _cl_event* cl_event

    ctypedef enum clblasOrder:
        clblasRowMajor
        clblasColumnMajor

    ctypedef enum clblasTranspose:
        clblasNoTrans
        clblasTrans
        clblasConjTrans

    clblasStatus clblasSetup()
    void clblasTeardown()

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


cdef get_status_message(clblasStatus status):
    if status == clblasSuccess:
        return "success"
    if status == clblasInvalidValue:
        return "invalid value"
    if status == clblasInvalidCommandQueue:
        return "invalid command queue"
    if status == clblasInvalidContext:
        return "invalid context"
    if status == clblasInvalidMemObject:
        return "invalid mem object"
    if status == clblasInvalidDevice:
        return "invalid device"
    if status == clblasInvalidEventWaitList:
        return "invalid event wait list"
    if status == clblasOutOfResources:
        return "out of resources"
    if status == clblasOutOfHostMemory:
        return "out of host memory"
    if status == clblasInvalidOperation:
        return "invalid operation"
    if status == clblasCompilerNotAvailable:
        return "compiler not available"
    if status == clblasBuildProgramFailure:
        return "build program failure"
    if status == clblasNotImplemented:
        return "clBLAS: not implemented"
    if status == clblasNotInitialized:
        return "clBLAS: not initialized"
    if status == clblasInvalidMatA:
        return "clBLAS: invalid mat A"
    if status == clblasInvalidMatB:
        return "clBLAS: invalid mat B"
    if status == clblasInvalidMatC:
        return "clBLAS: invalid mat C"
    if status == clblasInvalidVecX:
        return "clBLAS: invalid vec X"
    if status == clblasInvalidVecY:
        return "clBLAS: invalid vec Y"
    if status == clblasInvalidDim:
        return "clBLAS: invalid dim"
    if status == clblasInvalidLeadDimA:
        return "clBLAS: invalid lead dim A"
    if status == clblasInvalidLeadDimB:
        return "clBLAS: invalid lead dim B"
    if status == clblasInvalidLeadDimC:
        return "clBLAS: invalid lead dim C"
    if status == clblasInvalidIncX:
        return "clBLAS: invalid inc X"
    if status == clblasInvalidIncY:
        return "clBLAS: invalid inc Y"
    if status == clblasInsufficientMemMatA:
        return "clBLAS: insufficient mem mat A"
    if status == clblasInsufficientMemMatB:
        return "clBLAS: insufficient mem mat B"
    if status == clblasInsufficientMemMatC:
        return "clBLAS: insufficient mem mat C"
    if status == clblasInsufficientMemVecX:
        return "clBLAS: insufficient mem vec X"
    if status == clblasInsufficientMemVecY:
        return "clBLAS: insufficient mem vec Y"
    return "unrecognized status (code %d)" % status


def setup():
    cdef clblasStatus err = clblasSetup()
    if err != clblasSuccess:
        raise RuntimeError("Failed to setup clBLAS (Error %d)" % err)


def teardown():
    clblasTeardown()


def gemv(queue, A, X, Y, bool transA=False, float alpha=1.0, float beta=0.0):
    """Y <- beta * Y + alpha * dot(AT, X)
    where AT = A.T if transA else A
    """
    assert isinstance(A, Array) and len(A.shape) == 2 and A.dtype == 'float32'
    assert isinstance(X, Array) and len(X.shape) == 1 and X.dtype == 'float32'
    assert isinstance(Y, Array) and len(Y.shape) == 1 and Y.dtype == 'float32'

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]
    assert X.shape[0] == N
    assert Y.shape[0] == M

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
