# C_package/setup_fast_subgraphs.py

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "fast_subgraphs",
        ["fast_subgraphs.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="fast_subgraphs",
    version="0.1",
    ext_modules=ext_modules,
)