import hashlib
import os

import numpy as np
import pytest

import pyopencl_blas

maxint = np.iinfo(np.int32).max
test_seed = 0  # changing this will change seeds for all tests


def function_seed(function, mod=0):
    c = function.__code__

    # get function file path relative to project directory root
    # (so seed does not change across machines)
    project_path = os.path.abspath(os.path.dirname(pyopencl_blas.__file__))
    path = os.path.relpath(c.co_filename, start=project_path)

    # take start of md5 hash of function file and name, should be unique
    hash_list = os.path.normpath(path).split(os.path.sep) + [c.co_name]
    hash_string = bytes('/'.join(hash_list))
    i = int(hashlib.md5(hash_string).hexdigest()[:15], 16)
    return (i + mod) % maxint


@pytest.fixture
def rng(request):
    """a seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    seed = function_seed(request.function, mod=test_seed)
    return np.random.RandomState(seed)
