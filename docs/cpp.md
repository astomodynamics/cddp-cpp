# C++ build

The repository remains a first-class C++ library even when distributed through
PyPI for Python users.

## Core build

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## Python bindings from CMake

```bash
cmake -S . -B build-python -DCDDP_CPP_BUILD_PYTHON=ON -DCDDP_CPP_BUILD_TESTS=OFF -DCDDP_CPP_BUILD_EXAMPLES=OFF
cmake --build build-python -j4
```

## Installed C++ assets

The install rules export:

- `libcddp` and CMake package metadata
- public headers under `include/cddp-cpp/`
- the Python extension when `CDDP_CPP_BUILD_PYTHON=ON`

That lets the repository serve both as:

- a source build for C++ consumers
- the native backend for the `pycddp` wheel
