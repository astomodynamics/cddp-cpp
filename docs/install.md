# Installation

## Python users

Install the published wheel from PyPI:

```bash
pip install pycddp
```

Target release support is:

- CPython 3.10-3.13
- Linux x86_64
- macOS x86_64 and arm64
- Windows x86_64

If your platform does not have a wheel yet, `pip` will fall back to the source
distribution and build locally.

## Source build prerequisites

- CMake 3.15+
- a C++17 compiler
- Eigen 3
- Python 3.10+ when building the bindings

Ubuntu example:

```bash
sudo apt-get install build-essential cmake libeigen3-dev
```

macOS example:

```bash
brew install cmake eigen
```

## Local developer install

For repository work:

```bash
uv venv .venv --python 3.12
uv sync
source .venv/bin/activate
pytest -q python/tests
```

## Build the Python wheel locally

```bash
uv build
```

This creates `dist/*.tar.gz` and `dist/*.whl`.
