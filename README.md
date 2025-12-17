# An `everest-optimizers` optimizer plugin for `ropt`
This package installs a plugin for the
[`ropt`](https://github.com/TNO-ropt/ropt) robust optimization package,
providing access to algorithms provided by the `everest-optimizers` optimization
package.

`ropt-eo` is developed by the Netherlands Organisation for Applied Scientific
Research (TNO). All files in this repository are released under the GNU General
Public License v3.0 (a copy is provided in the LICENSE file).

See also the online [`ropt`](https://tno-ropt.github.io/ropt/) and
[`ropt-eo`](https://tno-ropt.github.io/ropt-eo/) manuals for more
information.


## Dependencies
This code has been tested with Python versions 3.11-3.13.

The plugin depends on the
[everest-optimizers](https://github.com/equinor/everest-optimizers) Python
wrappers of various optimization algorithms.


## Installation
From PyPI:
```bash
pip install ropt-eo
```


## Development
The `ropt-eo` source distribution can be found on
[GitHub](https://github.com/tno-ropt/ropt-eo). It uses a standard
`pyproject.toml` file, which contains build information and configuration
settings for various tools. A development environment can be set up with
compatible tools of your choice.

The `ropt-eo` package uses [ruff](https://docs.astral.sh/ruff/) (for formatting
and linting), [mypy](https://www.mypy-lang.org/) (for static typing), and
[pytest](https://docs.pytest.org/en/stable/) (for running the test suite).
