from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    packages=find_packages('emd'),
    ext_modules=[
        CUDAExtension('emd.cuda', [
            'emd/cuda/emd.cpp',
            'emd/cuda/emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })