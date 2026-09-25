---
id: packaging
title: Packaging and Release
sidebar_label: Packaging
description: PyPI metadata, setuptools src layout, wheel/sdist contents, dependencies, version synchronization, and local release gates.
---

# Packaging and release

## Distribution identity

```text
name: speedtronic
version: 2.0.0
requires-python: >=3.10
license: Apache-2.0
```

The distribution and import package are both named `speedtronic`. The repository URL may use title case, but PyPI normalization and metadata are exactly `speedtronic`.

## Build system

```toml
[build-system]
requires = ["setuptools>=77.0.3", "wheel"]
build-backend = "setuptools.build_meta"
```

The modern SPDX license expression and `license-files = ["LICENSE"]` require a sufficiently recent setuptools backend.

## Runtime dependencies

```text
torch>=2.1
safetensors>=0.4
PyYAML>=6.0
numpy>=1.24
huggingface-hub>=0.23
```

Install the appropriate CPU or CUDA PyTorch wheel for the target environment when the default PyPI wheel is not appropriate.

## Optional dependencies

### Development

```text
pytest>=7.4
ruff>=0.5
```

### Logging

```text
wandb>=0.16
tensorboard>=2.14
```

### Release

```text
build>=1.2
check-wheel-contents>=0.6
twine>=6.0
```

The release extra is for maintainers and is not required at runtime.

## Console entry point

```toml
[project.scripts]
speedtronic = "speedtronic.cli:main"
```

This creates:

```bash
speedtronic ...
```

The package also supports:

```bash
python -m speedtronic ...
```

## Source layout

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["speedtronic*"]
namespaces = false
```

The wheel contains only regular packages matching `speedtronic*`:

```text
speedtronic/
speedtronic/distributed/
```

Tests, configs, and examples are not installed as wheel package data.

## Source distribution manifest

`MANIFEST.in` includes:

```text
LICENSE
README.md
pyproject.toml
configs/*.yaml
examples/*.py
examples/*.yaml
CHANGELOG.md
tests/*.py
```

It excludes bytecode and generated run/state directories.

## Wheel versus sdist

| Artifact | Intended use | Includes |
|---|---|---|
| Wheel | Installation | Importable package, metadata, license, entry point |
| Source archive | Source review/build | Wheel source plus tests, configs, examples, manifest, README, license |

Commands such as `speedtronic train --config configs/smoke.yaml` are source-checkout/source-archive examples. An arbitrary installed wheel does not guarantee that top-level `configs/` exists beside the environment.

## Version synchronization

The version appears in:

```text
pyproject.toml
src/speedtronic/__init__.py
```

They currently both contain `2.0.0`. A release should update both and rebuild.

## Local release sequence

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m compileall src

python -m pip install build twine check-wheel-contents
rm -rf dist build
python -m build
python -m twine check dist/*
check-wheel-contents dist/speedtronic-2.0.0-py3-none-any.whl
```

Install into a clean environment and test the console command before upload.

## Artifact inspection

For a wheel:

- verify `Name: speedtronic`;
- verify version and Python requirement;
- verify Apache license file;
- verify console entry point;
- verify `speedtronic` and `speedtronic.distributed` modules;
- reject unexpected top-level packages.

For an sdist:

- inspect `SOURCES.txt`/archive contents;
- ensure tests/configs/examples are included;
- ensure virtual environments, runs, checkpoints, and bytecode are absent.

## PyPI upload

The project is published at:

```text
https://pypi.org/project/speedtronic/
```

Do not print or inspect credential files. Let Twine resolve its existing non-interactive configuration or SDK credential environment. A PyPI release filename cannot be overwritten once uploaded; fix metadata and increment the version for a new release.

## Documentation build

The Docusaurus project is isolated under `docs-site` and does not affect the Python wheel.

```bash
cd docs-site
npm ci
npm run build
npm run serve
```

Generated directories:

```text
docs-site/node_modules/
docs-site/.docusaurus/
docs-site/build/
```

are ignored locally.

## Release checklist

- [ ] Version values agree.
- [ ] Changelog/release intent is defined outside the source if needed.
- [ ] Tests, lint, formatting, and compilation pass.
- [ ] Wheel and sdist build.
- [ ] Twine metadata check passes.
- [ ] Wheel contents are clean.
- [ ] Clean-environment installation works.
- [ ] CLI and smoke run work from the wheel/source archive.
- [ ] No secret is stored in source or distribution metadata.
- [ ] Documentation build passes without deployment.

Related: [Testing](./testing), [Shipped Examples](../examples/shipped-examples), and [Troubleshooting](./troubleshooting).
