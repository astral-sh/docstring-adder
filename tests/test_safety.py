from __future__ import annotations

import ast
from pathlib import Path

import pytest

import add_docstrings


def test_assert_asts_match_allows_added_docstrings() -> None:
    """Docstrings may be inserted into ordinary and else bodies."""
    previous = ast.parse(
        '''\
"""Existing module docs."""
import sys

class C:
    value: int

def f(value: int) -> str: ...
if sys.platform == "win32":
    def platform_name() -> str: ...
else:
    def platform_name() -> str: ...
'''
    )
    new = ast.parse(
        '''\
"""Existing module docs."""
import sys

class C:
    """Class docs."""
    value: int

def f(value: int) -> str:
    """Function docs."""
if sys.platform == "win32":
    def platform_name() -> str:
        """Return the platform name."""
else:
    def platform_name() -> str:
        """Return the platform name."""
'''
    )

    add_docstrings.assert_asts_match(previous, new)


def test_assert_asts_match_rejects_different_node_types() -> None:
    """A statement cannot be replaced by a different kind of statement."""
    previous = ast.parse("value: int").body[0]
    new = ast.parse("def value(arg: int) -> None: ...").body[0]

    with pytest.raises(
        RuntimeError, match=r"AST node types differ: AnnAssign .* != FunctionDef"
    ):
        add_docstrings.assert_asts_match(previous, new)


def test_assert_asts_match_rejects_removed_body_nodes() -> None:
    """Existing statements cannot be removed from a suite."""
    previous = ast.parse("value: int\nother: str")
    new = ast.parse("value: int")

    with pytest.raises(RuntimeError, match="Nodes appear to have been removed"):
        add_docstrings.assert_asts_match(previous, new)


@pytest.mark.parametrize(
    ("previous", "new", "message"),
    [
        (
            "from typing import Literal\nVERSION: Literal[1]",
            'from typing import Literal\nVERSION: Literal["1"]',
            r"AST node types differ: int .* != str",
        ),
        (
            "from typing import Literal\nVERSION: Literal[1]",
            "from typing import Literal\nVERSION: Literal[2]",
            r"AST node values differ: 1 .* != 2",
        ),
        ("value: int", "value: str", r"AST node values differ: int .* != str"),
    ],
)
def test_assert_asts_match_rejects_changed_fields(
    previous: str, new: str, message: str
) -> None:
    """Scalar AST fields must retain both their types and values."""
    with pytest.raises(RuntimeError, match=message):
        add_docstrings.assert_asts_match(ast.parse(previous), ast.parse(new))


def test_assert_asts_match_rejects_changed_parameter_count() -> None:
    """A function signature cannot gain an additional parameter."""
    previous = ast.parse("def f(value: int) -> str: ...")
    new = ast.parse("def f(value: int, other: str) -> str: ...")

    with pytest.raises(RuntimeError, match="AST node lists differ in length: 1 != 2"):
        add_docstrings.assert_asts_match(previous, new)


def test_assert_asts_match_rejects_added_attribute_initializer() -> None:
    """An annotated stub attribute cannot gain an initializer."""
    previous = ast.parse("value: int")
    new = ast.parse("value: int = ...")

    with pytest.raises(
        RuntimeError, match=r"AST node values differ in type: NoneType .* != Constant"
    ):
        add_docstrings.assert_asts_match(previous, new)


def test_check_no_destructive_changes_accepts_added_docstring(tmp_path: Path) -> None:
    """The top-level safety check accepts a permitted docstring insertion."""
    add_docstrings.check_no_destructive_changes(
        tmp_path / "module.pyi",
        "def f(value: int) -> str: ...\n",
        'def f(value: int) -> str:\n    """Function docs."""\n',
    )


def test_check_no_destructive_changes_reports_changed_ast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A destructive AST change is reported before its error is reraised."""
    path = tmp_path / "module.pyi"

    with pytest.raises(RuntimeError):
        add_docstrings.check_no_destructive_changes(
            path, "value: int\n", "value: str\n"
        )

    assert (
        f"ERROR: new stub file at {path} has destructive changes"
        in capsys.readouterr().out
    )
