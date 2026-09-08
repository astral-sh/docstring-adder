from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import add_docstrings


def test_load_blacklist_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    """Blacklist files support comments, whitespace, and duplicate entries."""

    blacklist = tmp_path / "blacklist.txt"
    blacklist.write_text(
        """\
        pathlib.Path.owner  # unavailable on some platforms

        urllib3.PoolManager.request
        urllib3.PoolManager.request
        # Explanatory comment
        """,
        encoding="utf-8",
    )

    assert add_docstrings.load_blacklist(blacklist) == frozenset({
        "pathlib.Path.owner",
        "urllib3.PoolManager.request",
    })


def test_load_module_overrides_parses_table(tmp_path: Path) -> None:
    """A module-overrides table maps stub module names to runtime modules."""

    config = tmp_path / "pyproject.toml"
    config.write_text(
        "[tool.docstring-adder.module-overrides]\n"
        '"example_package.submodule" = "example_package._runtime"\n'
        '"example_package.other_submodule" = "example_package._runtime"\n',
        encoding="utf-8",
    )

    assert add_docstrings.load_module_overrides(config) == {
        "example_package.submodule": "example_package._runtime",
        "example_package.other_submodule": "example_package._runtime",
    }


def test_load_module_overrides_rejects_non_string_value(tmp_path: Path) -> None:
    """A non-string override value fails loudly rather than being coerced."""

    config = tmp_path / "pyproject.toml"
    config.write_text(
        "[tool.docstring-adder.module-overrides]\n"
        '"example_package.submodule" = ["example_package._runtime"]\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        add_docstrings.load_module_overrides(config)


def test_discover_module_overrides_reads_nearby_pyproject(tmp_path: Path) -> None:
    """Overrides are auto-discovered from a nearby pyproject.toml."""

    distribution = tmp_path / "python" / "example_package-stubs"
    package = distribution / "example_package"
    package.mkdir(parents=True)
    (tmp_path / "python" / "pyproject.toml").write_text(
        "[tool.docstring-adder.module-overrides]\n"
        '"example_package.submodule" = "example_package._runtime"\n',
        encoding="utf-8",
    )

    assert add_docstrings.discover_module_overrides([package]) == {
        "example_package.submodule": "example_package._runtime"
    }


def _write_override_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    """Build an importable runtime module with one documented function."""

    (tmp_path / f"{module_name}.py").write_text(
        'def f() -> None:\n    """Override docs."""\n', encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))


def test_main_applies_module_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override redirects introspection to a different runtime module."""

    runtime_module = "example_package_runtime"
    _write_override_target(tmp_path, monkeypatch, runtime_module)

    package = tmp_path / "stubs" / "example_package"
    package.mkdir(parents=True)
    stub = package / "submodule.pyi"
    stub.write_text("def f() -> None: ...\n", encoding="utf-8")

    (package.parent / "pyproject.toml").write_text(
        "[tool.docstring-adder.module-overrides]\n"
        f'"example_package.submodule" = "{runtime_module}"\n',
        encoding="utf-8",
    )

    args = ["--packages", str(package)]
    with (
        patch.object(
            add_docstrings.typeshed_client,  # type: ignore[attr-defined]
            "get_all_stub_files",
            return_value=[("example_package.submodule", stub)],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        add_docstrings._main(args)

    assert exc_info.value.code == 0
    assert '"""Override docs."""' in stub.read_text(encoding="utf-8")


def test_main_applies_module_override_to_typeshed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overrides are also discovered for --typeshed-packages roots."""

    runtime_module = "example_package_typeshed_runtime"
    _write_override_target(tmp_path, monkeypatch, runtime_module)

    package = tmp_path / "typeshed-stubs" / "example_package"
    package.mkdir(parents=True)
    stub = package / "submodule.pyi"
    stub.write_text("def f() -> None: ...\n", encoding="utf-8")

    (package.parent / "pyproject.toml").write_text(
        "[tool.docstring-adder.module-overrides]\n"
        f'"example_package.submodule" = "{runtime_module}"\n',
        encoding="utf-8",
    )

    args = ["--typeshed-packages", str(package)]
    with (
        patch.object(add_docstrings, "install_typeshed_packages"),
        patch.object(
            add_docstrings.typeshed_client,  # type: ignore[attr-defined]
            "get_all_stub_files",
            return_value=[("example_package.submodule", stub)],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        add_docstrings._main(args)

    assert exc_info.value.code == 0
    assert '"""Override docs."""' in stub.read_text(encoding="utf-8")


def test_install_typeshed_packages_rejects_a_directory_without_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A package directory must contain typeshed's METADATA.toml file."""

    package = tmp_path / "urllib3"
    package.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        add_docstrings.install_typeshed_packages([package])

    assert exc_info.value.code == 1
    assert f"{package} does not look like a typeshed package" in capsys.readouterr().err


def test_main_exits_when_no_matching_stubs_are_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI reports an error when discovery finds no requested package stubs."""

    package = tmp_path / "stubs" / "urllib3"
    package.mkdir(parents=True)
    unrelated_stub = tmp_path / "other" / "unrelated.pyi"
    args = ["--packages", str(package)]

    with (
        patch.object(
            add_docstrings.typeshed_client,  # type: ignore[attr-defined]
            "get_all_stub_files",
            return_value=[("unrelated", unrelated_stub)],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        add_docstrings._main(args)

    assert exc_info.value.code == 1
    assert "Didn't find any stubs to codemod" in capsys.readouterr().out


def test_main_rejects_invalid_stdlib_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stdlib path must be a directory containing a VERSIONS file."""

    invalid_stdlib = tmp_path / "stdlib"
    invalid_stdlib.mkdir()
    args = ["--stdlib-path", str(invalid_stdlib)]

    with pytest.raises(SystemExit) as exc_info:
        add_docstrings._main(args)

    assert exc_info.value.code == 2
    assert (
        f'"{invalid_stdlib}" does not point to a valid stdlib directory'
        in capsys.readouterr().err
    )
