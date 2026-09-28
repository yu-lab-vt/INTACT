from setuptools import setup, Extension
import pybind11
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args = []

if sys.platform != "win32":
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]

sources = [
    "fast_full_fusion.cpp",
    "cinda_funcs.c",
    "edt_3d.cpp",
]

ext_modules = [
    Extension(
        "fast_full_fusion",
        sources=sources,
        include_dirs=[
            pybind11.get_include(),
            this_dir,
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="fast_full_fusion",
    version="0.1",
    ext_modules=ext_modules,
)