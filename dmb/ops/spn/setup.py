from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='gaterecurrent2dnoind_cuda',
    ext_modules=[
        CUDAExtension('gaterecurrent2dnoind_cuda', [
            'src/gaterecurrent2dnoind_cuda.cpp',
            'src/gaterecurrent2dnoind_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


