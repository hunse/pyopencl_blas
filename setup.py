from setuptools import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = []
ext_modules.append(Extension(
    'pyopencl_blas.blas',
    ['pyopencl_blas/blas.pyx'],
    include_dirs=[
        '/opt/AMDAPP/include',
        '/opt/clBLAS-2.4.0-Linux-x64/include'],
    library_dirs=['/opt/clBLAS-2.4.0-Linux-x64/lib64'],
    libraries=['clBLAS'],
    language='c++',
    extra_compile_args=['-w', '-O3']))

setup(
    name='pyopencl_blas',
    version='0.0.1',
    author="Eric Hunsberger",
    author_email="erichuns@gmail.com",
    url="https://github.com/hunse/pyopencl_blas",
    license="MIT",
    description="PyOpenCL wrapper for AMD clMathLibraries.clBLAS",
    requires=[
        'pyopencl',
        'cython',
    ],
    packages=['pyopencl_blas'],
    scripts=[],
    ext_modules=ext_modules,
    cmdclass = {'build_ext': build_ext},
)
