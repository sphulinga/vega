# Releasing to PyPI

PyVega wheels contain the compiled session server as a CPython extension
(`pyvega/_vega_server.*.so`), so `pip install pyvega` is all a user needs --
no CMake, no separate `vega_server` binary. Releases are built and published by
[`.github/workflows/release.yml`](../.github/workflows/release.yml).

## What gets built

| Artifact | Platforms |
|----------|-----------|
| sdist (`pyvega-X.Y.Z.tar.gz`) | builds anywhere with CMake >= 3.16 and a C++17 compiler |
| wheels | CPython 3.9 - 3.13 on manylinux_2_28 x86_64 / aarch64, macOS 11+ arm64 / x86_64 |

Every wheel is smoke-tested by cibuildwheel (`pytest python/tests`) in a clean
environment before anything is uploaded. Windows is not supported (POSIX
sockets).

## One-time setup (repository admin)

1. **Register the trusted publisher on PyPI.** `pyvega` already exists on
   PyPI (0.5.0 was uploaded by hand), so add it to the existing project:
   <https://pypi.org/manage/project/pyvega/settings/publishing/> (*pyvega ->
   Manage -> Publishing*), signed in as a project owner. Add a GitHub publisher
   with exactly:
   - Owner: `sphulinga`
   - Repository: `vega`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   (For a project that does not exist yet, add the same values as a *pending
   publisher* on <https://pypi.org/manage/account/publishing/> instead.)
2. Do the same on TestPyPI (<https://test.pypi.org>) with environment
   `testpypi`, if you want manual dry runs.
3. In GitHub, *Settings -> Environments*, make sure environments `pypi` and
   `testpypi` exist. Adding required reviewers to `pypi` gives a manual
   approval gate before each upload.

If the publish job fails with `invalid-publisher` ("no corresponding
publisher"), step 1 is missing or one of its four values doesn't match. The
job log prints the claims GitHub sent (`repository`, `workflow_ref`,
`environment`); compare them with the publisher on PyPI. Nothing is uploaded
in that case, so after fixing it use *Re-run failed jobs* on the same run.

No API tokens or secrets are needed: the workflow authenticates with OIDC
(`id-token: write`).

## Cutting a release

1. Bump `__version__` in `python/pyvega/__init__.py` -- the **only** place the
   version lives (CMake and the package metadata read it from there).
2. Move the `Unreleased` notes in `CHANGELOG.md` under the new version.
3. Optional dry run: *Actions -> Release (PyPI) -> Run workflow* with
   `repository = testpypi`, then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyvega==X.Y.Z`.
4. Merge to `main`, then tag and push:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

   The tag may be `vX.Y.Z` or `X.Y.Z`; push only one of them per release
   (each tag push starts a separate release run).

   The workflow checks that the tag matches `__version__`, builds the sdist and
   all wheels, publishes to PyPI, and creates a GitHub Release with the files
   attached.

## Building locally

```bash
pip install build
python -m build                        # sdist + wheel for this machine
pip install dist/pyvega-*.whl
python -c "from pyvega import SparkContext; print(SparkContext(2).range(0, 10).sum())"
```

To reproduce the release matrix locally (needs Docker on Linux):

```bash
pip install cibuildwheel
cibuildwheel --platform linux
```
