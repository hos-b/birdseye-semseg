# https://gist.github.com/tonyseek/7821993
import cv2
import glob
from os import path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

abs_path = osp.dirname(osp.realpath(__file__))
extra_objects = [osp.join(abs_path, 'build/lie_so3.so')]
extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')


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