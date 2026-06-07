"""Tests for the ASTTokens source transformer."""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import pytest
import typeshed_client

from add_docstrings import SourceEditor, TextOffset, transform_stub_source


class DocumentedValue:
    """Runtime value whose instance docstring is useful for a stub attribute."""


CONTEXT = typeshed_client.get_search_context(
    version=(sys.version_info.major, sys.version_info.minor), platform=sys.platform
)


def transform(source: str, runtime_module: types.ModuleType) -> str:
    """Transform a synthetic stub using a synthetic runtime module."""
    return transform_stub_source(
        source,
        module_name=runtime_module.__name__,
        runtime_module=runtime_module,
        stub_file_path=Path(f"{runtime_module.__name__}.pyi"),
        typeshed_client_context=CONTEXT,
        blacklisted_objects=frozenset(),
    )


def runtime_module() -> types.ModuleType:
    """Return a runtime module with documented definitions and attributes."""
    module = types.ModuleType("test_module")

    def f() -> None:
        """Function docs."""

    def g() -> None:
        """Other function docs."""

    class C:
        def method(self) -> None:
            """Method docs."""

    value = DocumentedValue()
    value.__doc__ = "Attribute docs."
    module.__dict__.update(f=f, g=g, C=C, x=value, y=value, z=value)
    type.__setattr__(C, "x", value)
    return module


def test_inline_ellipsis_preserves_trailing_comment() -> None:
    """An inline ellipsis becomes a block without losing its comment."""
    source = textwrap.dedent(
        """\
        def f(): ...  # type: ignore[misc]
        """
    )
    expected = textwrap.dedent(
        '''\
        def f():  # type: ignore[misc]
            """Function docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_inline_ellipsis_with_semicolon() -> None:
    """A trailing semicolon is removed when replacing an inline ellipsis."""
    source = textwrap.dedent(
        """\
        def f(): ...;
        """
    )
    expected = textwrap.dedent(
        '''\
        def f():
            """Function docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_inline_ellipsis_with_semicolon_and_comment() -> None:
    """Removing a semicolon preserves the old comment attachment behavior."""
    source = textwrap.dedent(
        """\
        def f(): ...;  # comment
        """
    )
    expected = textwrap.dedent(
        '''\
        def f():# comment
            """Function docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_inline_ellipsis_without_final_newline() -> None:
    """Adding a definition docstring does not add a final newline."""
    source = "def f(): ..."
    expected = textwrap.dedent(
        '''\
        def f():
            """Function docs."""'''
    )
    assert transform(source, runtime_module()) == expected


def test_block_body_preserves_comments_and_ellipsis() -> None:
    """Block comments and the existing multiline ellipsis remain untouched."""
    source = textwrap.dedent(
        """\
        def f():
            # before
            ...  # after
        """
    )
    expected = textwrap.dedent(
        '''\
        def f():
            """Function docs."""
            # before
            ...  # after
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_only_first_overload_is_documented() -> None:
    """Duplicate definitions in one suite retain the overload behavior."""
    source = textwrap.dedent(
        """\
        def f(x: int): ...
        def f(x: str): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        def f(x: int):
            """Function docs."""
        def f(x: str): ...
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_nested_method_is_resolved_from_runtime_class() -> None:
    """Methods are looked up through the runtime class namespace."""
    source = textwrap.dedent(
        """\
        class C:
            def method(self): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        class C:
            def method(self):
                """Method docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_missing_runtime_class_is_unchanged() -> None:
    """No docstrings are retrieved through the missing-runtime sentinel."""
    source = textwrap.dedent(
        """\
        class Missing:
            __dict__: object
            class __class__: ...
            class __dict__: ...
        """
    )
    assert transform(source, runtime_module()) == source


def test_unreachable_branch_is_not_documented() -> None:
    """Only the reachable arm of a version check receives docstrings."""
    source = textwrap.dedent(
        """\
        import sys
        if sys.version_info >= (3, 0):
            def f(): ...
        else:
            def g(): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        import sys
        if sys.version_info >= (3, 0):
            def f():
                """Function docs."""
        else:
            def g(): ...
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_attribute_docstrings_preserve_comments_and_spacing() -> None:
    """Attribute docs keep comments and existing blank lines in place."""
    source = textwrap.dedent(
        """\
        x: int
        # about y
        y: str

        z: bytes
        """
    )
    expected = textwrap.dedent(
        '''\
        x: int
        """Attribute docs."""

        # about y
        y: str
        """Attribute docs."""

        z: bytes
        """Attribute docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_semicolon_separated_attributes_are_unchanged() -> None:
    """Multiple assignments on one physical line are not documented."""
    source = textwrap.dedent(
        """\
        x: int; y: str
        """
    )
    assert transform(source, runtime_module()) == source


def test_attribute_without_final_newline() -> None:
    """Adding an attribute docstring does not add a final newline."""
    source = "x: int"
    expected = textwrap.dedent(
        '''\
        x: int
        """Attribute docs."""'''
    )
    assert transform(source, runtime_module()) == expected


def test_inserted_blank_line_keeps_suite_indentation() -> None:
    """Blank lines added in indented suites retain indentation spaces."""
    source = textwrap.dedent(
        """\
        class C:
            x: int
            pass
        """
    )
    # Keep the indentation-only blank line explicit: dedent strips its spaces.
    expected = 'class C:\n    x: int\n    """Attribute docs."""\n    \n    pass\n'
    assert transform(source, runtime_module()) == expected


def test_existing_docstring_is_preserved() -> None:
    """An existing definition docstring is not replaced."""
    source = textwrap.dedent(
        '''\
        def f():
            """Stub docs."""
        '''
    )
    assert transform(source, runtime_module()) == source


def test_tabs_are_preserved() -> None:
    """Inserted block docstrings use the source file's indentation style."""
    source = """def f():
\t...
"""
    expected = '''def f():
\t"""Function docs."""
\t...
'''
    assert transform(source, runtime_module()) == expected


def test_multiline_header_comment_is_preserved() -> None:
    """A comment after a multiline signature stays on the header."""
    source = textwrap.dedent(
        """\
        def f(
            x: int,
        ):  # header
            ...
        """
    )
    expected = textwrap.dedent(
        '''\
        def f(
            x: int,
        ):  # header
            """Function docs."""
            ...
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_reachable_elif_is_documented() -> None:
    """Reachability tracks a real elif as part of the same branch chain."""
    source = textwrap.dedent(
        """\
        import sys
        if sys.version_info < (3, 0):
            def g(): ...
        elif sys.version_info >= (3, 0):
            def f(): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        import sys
        if sys.version_info < (3, 0):
            def g(): ...
        elif sys.version_info >= (3, 0):
            def f():
                """Function docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_reachable_else_if_is_documented() -> None:
    """A nested `if` in an `else` suite is visited as a separate suite."""
    source = textwrap.dedent(
        """\
        import sys
        if sys.version_info < (3, 0):
            def g(): ...
        else:
            if sys.version_info >= (3, 0):
                def f(): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        import sys
        if sys.version_info < (3, 0):
            def g(): ...
        else:
            if sys.version_info >= (3, 0):
                def f():
                    """Function docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_multiline_attribute_docstring_in_else_if_is_indented() -> None:
    """A physical `else` suite contributes to attribute-docstring indentation."""
    module = runtime_module()
    module.x.__doc__ = "First line.\nSecond line."
    source = textwrap.dedent(
        """\
        import sys
        if sys.version_info < (3, 0):
            pass
        else:
            if sys.version_info >= (3, 0):
                x: int
        """
    )
    expected = textwrap.dedent(
        '''\
        import sys
        if sys.version_info < (3, 0):
            pass
        else:
            if sys.version_info >= (3, 0):
                x: int
                """First line.
                Second line.
                """
        '''
    )
    assert transform(source, module) == expected


def test_attribute_docstrings_in_generic_compound_suites() -> None:
    """Attribute docs are added in every physical indented suite."""
    source = textwrap.dedent(
        """\
        try:
            x: int
        except Exception:
            y: str
        finally:
            z: bytes
        """
    )
    expected = textwrap.dedent(
        '''\
        try:
            x: int
            """Attribute docs."""
        except Exception:
            y: str
            """Attribute docs."""
        finally:
            z: bytes
            """Attribute docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_attribute_docstrings_in_match_cases() -> None:
    """Attribute docs are added in suites nested inside `match_case` nodes."""
    source = textwrap.dedent(
        """\
        match object():
            case int():
                x: int
            case _:
                y: str
        """
    )
    expected = textwrap.dedent(
        '''\
        match object():
            case int():
                x: int
                """Attribute docs."""
            case _:
                y: str
                """Attribute docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_non_ascii_prefix_does_not_shift_edits() -> None:
    """AST byte offsets do not corrupt character-based source edits."""
    source = textwrap.dedent(
        """\
        \N{GREEK SMALL LETTER PI}: int
        def f(): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        \N{GREEK SMALL LETTER PI}: int
        def f():
            """Function docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_transform_is_idempotent() -> None:
    """Running the source transformer twice does not change the result."""
    module = runtime_module()
    source = textwrap.dedent(
        """\
        def f(): ...
        x: int
        """
    )
    transformed = transform(source, module)
    assert transform(transformed, module) == transformed


def test_conflicting_edits_have_an_actionable_error() -> None:
    """Conflicting edits report the ranges that indicate an editor bug."""
    editor = SourceEditor("abcdef")
    editor.replace(TextOffset(1), TextOffset(4), "first")
    editor.replace(TextOffset(2), TextOffset(5), "second")

    with pytest.raises(
        RuntimeError,
        match=r"SourceEdit\(start=1, end=4.*overlaps SourceEdit\(start=2, end=5",
    ):
        editor.render()
