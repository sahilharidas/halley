# Releasing

A release is a tag. Pushing `vX.Y.Z` to GitHub runs `.github/workflows/release.yml`, which checks
the tagged commit the way the development checks do, builds the sdist and the wheel, installs the
wheel into a clean environment and runs the test suite against it, then uploads both files to
PyPI. The upload uses PyPI's trusted publishing: PyPI accepts GitHub's signed statement that the
request comes from this repository's `release.yml` running in its `pypi` environment, so no API
token exists anywhere, in the repository or outside it.

## Cutting a release

1. Bump the version in `pyproject.toml` and in `src/halley/__init__.py` together. The site's title
   reads `halley.__version__`, and `tests/test_skeleton.py` checks that it agrees with the
   installed package's metadata.
2. Run `uv lock`, so that `uv.lock` records the new version; `uv sync --locked` refuses a
   lockfile that is out of date.
3. Commit, tag the commit `vX.Y.Z` and push the tag:

   ```bash
   git commit -am "Release X.Y.Z"
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

The workflow stops before building if the tag's version differs from the version in
`pyproject.toml`. PyPI accepts each version once: a run that failed before the upload can be
repeated from the Actions tab, but a published version cannot be replaced, so a mistake is fixed in
the next version.

## What the workflow does

- **check**: on the Python version in `.python-version`, `uv sync --locked`, `ruff check`,
  `ruff format --check`, `pyright` and `pytest`, then `uv build --no-sources`.
- **smoke**: a clean virtual environment gets the runtime dependencies exported from `uv.lock`
  with their hashes, then the wheel with `--no-deps`, and `uv pip check` confirms that the wheel's
  requirements are met. The test suite then runs under `python -I` from a copy that holds only
  `pyproject.toml`, `tests/`, `docs/spec` and `docs/tutorial`, so the only `halley` it can import
  is the installed wheel.
- **publish**: `uv publish --trusted-publishing always` uploads the files the smoke test ran
  against. The job runs in the `pypi` environment, and its log prints the SHA-256 hash of each
  file, which PyPI shows on the release's file list.

The same checks run locally with the commands under Development in the README, followed by
`uv build --no-sources`.

## One-time setup

Trusted publishing is configured once, by the project's owner, in two places.

On PyPI, under *Manage* for the project `halley`, open *Publishing* and add a GitHub publisher with
owner `sahilharidas`, repository `halley`, workflow `release.yml` and environment `pypi`.

On GitHub, under the repository's *Settings* and *Environments*, create an environment named
`pypi`. It can be left open, or given required reviewers, in which case every release waits for
approval before the upload runs.
