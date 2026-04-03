# Python package

The Python package is published as `pycddp`.

## What is included

- the native extension module `_pycddp_core`
- the public package namespace `pycddp`
- version metadata from `pycddp._version`

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
