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
    ctypedef double cl_double
    ctypedef unsigned int cl_uint

    ctypedef struct cl_float2:
        cl_float x
        cl_float y

    ctypedef struct cl_double2:
        cl_double x
        cl_double y

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

    ctypedef enum clblasUplo:
        clblasUpper
        clblasLower

    ctypedef enum clblasTranspose:
        clblasNoTrans
        clblasTrans
        clblasConjTrans


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


# def check_array(a, ndim, dtype, name):
#     if not isinstance(a, Array):


# def check_matrix_ty
