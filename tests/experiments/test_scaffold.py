"""Verify the experiments package and its project configuration."""

import re
from pathlib import Path

import tomllib
from packaging.requirements import Requirement


def test_experiments_import():
    """Resolve experiments to the documented top-level package."""
    import experiments

    root = Path(__file__).resolve().parents[2]
    assert Path(experiments.__file__).resolve() == root / "experiments" / "__init__.py"
    assert experiments.__doc__ and experiments.__doc__.strip()


def test_experiments_dependencies():
    """Require exactly the intended experiments dependency names."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    requirements = project["project"]["optional-dependencies"]["experiments"]
    names = {
        re.sub(r"[-_.]+", "-", Requirement(req).name).lower()
        for req in requirements
    }
    assert names == {
        "pandas", "polars", "pyarrow", "openpyxl", "pytest-benchmark", "psutil"
    }


def test_wheel_packages():
    """Keep the experiments package outside the bopp wheel."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["bopp"]
