# Contributing

Thanks for contributing to `cddp-cpp`.

## Scope

This project contains:

- the core C++ trajectory optimization library
- a curated set of C++ reference examples
- optional Python bindings and portfolio demos

Keep changes focused. Avoid mixing bug fixes, refactors, formatting churn, and documentation cleanup in one pull request unless they are tightly coupled.

## Development setup

### C++ build

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

### C++ examples

```bash
cmake -S . -B build -DCDDP_CPP_BUILD_EXAMPLES=ON
cmake --build build --target cddp_pendulum cddp_cartpole cddp_unicycle cddp_quadrotor_point cddp_manipulator -j4
```

### Python bindings

This repository uses `uv` for Python environment management.

```bash
uv venv .venv --python 3.12
uv sync
source .venv/bin/activate
pytest -q python/tests
```

## Pull requests

Before opening a pull request:

1. Make the smallest reasonable change for the problem you are solving.
2. Add or update tests when behavior changes.
3. Run the relevant build and test commands for the area you touched.
4. Update documentation when user-facing behavior, examples, or public APIs change.
5. Write a clear commit message and PR description.

PRs that include a minimal reproduction, exact validation commands, and a concise explanation of the design tradeoffs are much easier to review.

## Issues

- Use bug reports for reproducible defects.
- Use feature requests for API additions, solver improvements, or workflow changes.
- For sensitive security issues, follow [SECURITY.md](SECURITY.md) instead of opening a detailed public issue.
