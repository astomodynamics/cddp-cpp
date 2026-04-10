# CDDP

`cddp-cpp` is a constrained differential dynamic programming solver library with:

- a C++17 core library for trajectory optimization and MPC
- `pycddp` Python bindings built with `pybind11`
- a small animation-oriented Python portfolio for demos and regression checks
- IPDDP support for path constraints plus terminal equality and terminal inequality constraints

Use the navigation to get started with installation, local development, and
the release workflow for PyPI and GitHub Pages.

## Project scope

- C++ library: reusable solver and dynamical-system implementations
- Python package: importable bindings distributed as `pycddp`
- Docs site: Markdown content from `docs/`, published to GitHub Pages

## Portfolio highlight

The Python portfolio page is the visual entry point for the bindings and their
animation-driven demos.

[Open the portfolio gallery](python_portfolio.md){ .md-button .md-button--primary }
[Read the Python package guide](python.md){ .md-button }

## Quick start

Install the Python package from PyPI:

```bash
pip install pycddp
```

Or build from source:

```bash
git clone https://github.com/astomodynamics/cddp-cpp
cd cddp-cpp
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```
