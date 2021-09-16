from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lie_so3',
    ext_modules=[
        CUDAExtension('lie_so3', [
            'src/lie_so3_module_cuda.cpp',
            'src/lie_so3.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
},
    package_data={
        'static': ['include/*'],
    }
)