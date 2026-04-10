# Python package

The Python package is published as `pycddp`.

## What is included

- the native extension module `_pycddp_core`
- the public package namespace `pycddp`
- version metadata from `pycddp._version`
- `CDDP` problem bindings with `add_constraint(...)` and `add_terminal_constraint(...)`

## Solver coverage

The Python package follows the same solver capability as the C++ core. For
`IPDDP`, the regression suite now covers:

- path-only constrained problems
- terminal-inequality-only problems
- terminal-equality-only problems
- mixed path + terminal-equality problems
- warm-start reuse of path, terminal-inequality, and terminal-equality state

The current Python portfolio focuses on animated demos rather than terminal
constraint examples, so terminal-constrained IPDDP coverage is documented here
and exercised in tests.

## Local validation

Run the Python tests after syncing the environment:

```bash
source .venv/bin/activate
pytest -q python/tests
```

Build an installable artifact:

```bash
uv build
```

Smoke-test the wheel in a fresh environment:

```bash
uv venv /tmp/pycddp-smoke --python 3.13
source /tmp/pycddp-smoke/bin/activate
uv pip install dist/*.whl
python -c "import pycddp; print(pycddp.__version__)"
```

## Release strategy

- pull requests and pushes validate wheel builds in GitHub Actions
- tagged releases build wheels plus an sdist and publish to PyPI
- publishing uses PyPI Trusted Publishing instead of a long-lived API token
