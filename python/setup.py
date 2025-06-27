import os
import sys
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

FFMPEG_INCLUDE_DIR = os.getenv('FFMPEG_INCLUDE_DIR', '/usr/include/ffmpeg')
FFMPEG_LIB_DIR = os.getenv('FFMPEG_LIB_DIR', '/usr/lib64')
OPENCV_INCLUDE_DIR = os.getenv('OPENCV_INCLUDE_DIR', '/usr/include/opencv4')
OPENCV_LIB_DIR = os.getenv('OPENCV_LIB_DIR', '/usr/lib64')

try:
    import pybind11
    PYBIND11_INCLUDE_DIR = pybind11.get_include()
except ImportError:
    print("pybind11 not found. Please install it: pip install pybind11", file=sys.stderr)
    sys.exit(1)

CXX_FLAGS = ['-std=c++17', '-O3', '-D_GNU_SOURCE', '-D_POSIX_C_SOURCE=200809L']

ext_modules = [
    Extension(
        'ffmpeg_video',
        sources=[os.path.join('src', 'ffmpeg_video.cpp'), os.path.join('src', 'bindings.cpp')], 
        include_dirs=[
            PYBIND11_INCLUDE_DIR,
            FFMPEG_INCLUDE_DIR,
            OPENCV_INCLUDE_DIR,
            'src', 
        ],
        library_dirs=[
            FFMPEG_LIB_DIR,
            OPENCV_LIB_DIR,
        ],
        libraries=[
            'avformat',
            'avcodec',
            'avutil',
            'avfilter',
            'opencv_core',
            'opencv_highgui',
            'opencv_imgproc',
        ],
        extra_compile_args=CXX_FLAGS,
        extra_link_args=[],
        language='c++'
    ),
]

setup(
    name='ffmpeg_video',
    version='0.1.0',
    author='Balint Cristian',
    description='Python bindings for FFMPEGVideo C++ class with RKMpp acceleration',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'numpy', 'opencv-python'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
