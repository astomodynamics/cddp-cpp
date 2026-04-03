# Release operations

This repository now has three distinct automation paths:

- `wheels.yml`: validates wheel and sdist builds on pull requests and pushes
- `publish.yml`: builds release artifacts from a Git tag and publishes to PyPI
- `pages.yml`: publishes the documentation site to GitHub Pages

## One-time PyPI setup

Before the first release, configure PyPI Trusted Publishing for project
`pycddp`.

On PyPI:

1. Create the project if it does not exist yet.
2. Add a trusted publisher for GitHub Actions.
3. Set owner to `astomodynamics`.
4. Set repository to `cddp-cpp`.
5. Set workflow to `.github/workflows/publish.yml`.
6. Set environment name to `pypi`.

After that, pushing a tag like `v0.1.0` will publish the matching artifacts.

## One-time GitHub Pages setup

On GitHub:

1. Open repository settings.
2. Open Pages.
3. Set the source to `GitHub Actions`.

After that, pushes to `master` that touch the docs or `mkdocs.yml` will deploy
the site.

## Release steps

1. Update package version if needed.
2. Merge the release changes to `master`.
3. Create and push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

4. Verify the `publish.yml` workflow succeeds.
5. Confirm the release appears on PyPI and the docs site is current.
