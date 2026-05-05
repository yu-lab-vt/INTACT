# Fast easy fusion extension

Copy these files into `/home/cyf/INTACT/INTACT/C_package`:

- `fast_easy_fusion.cpp`
- `setup_fast_easy_fusion.py`
- `fast_easy_fusion_wrapper.py`

Make sure the same directory also contains:

- `libedt3d.so`
- `lib_cinda_funcs.so`

Compile:

```bash
cd /home/cyf/INTACT/INTACT/C_package
rm -rf build
rm -f fast_easy_fusion*.so
/home/cyf/.conda/envs/Allen/bin/python setup_fast_easy_fusion.py build_ext --inplace
```

Test import:

```bash
cd /home/cyf/INTACT/INTACT/C_package
/home/cyf/.conda/envs/Allen/bin/python -c "import fast_easy_fusion; print(fast_easy_fusion.__file__); print('ok')"
```
