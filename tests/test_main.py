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
