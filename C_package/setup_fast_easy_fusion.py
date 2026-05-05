from setuptools import setup, Extension
import pybind11
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]
    extra_link_args = []
    libraries = []
else:
    extra_compile_args = ["-O3", "-std=c++17", "-fopenmp"]
    extra_link_args = ["-fopenmp", f"-Wl,-rpath,{this_dir}"]
    libraries = ["edt3d", "_cinda_funcs"]

cpp_path = os.path.join(this_dir, "fast_easy_fusion.cpp")
if not os.path.exists(cpp_path) or os.path.getsize(cpp_path) == 0:
    raise RuntimeError("fast_easy_fusion.cpp is missing or empty.")

ext_modules = [
    Extension(
        "fast_easy_fusion",
        [cpp_path],
        include_dirs=[pybind11.get_include()],
        library_dirs=[this_dir],
        libraries=libraries,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(name="fast_easy_fusion", version="0.1", ext_modules=ext_modules)
