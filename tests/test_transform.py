from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import typeshed_client

import add_docstrings


def transform(source: str, runtime_module: types.ModuleType) -> str:
    """Add docstrings to the source as if it were a stub for the runtime module."""
    typeshed_client_context = typeshed_client.get_search_context(
        version=sys.version_info[:2], platform=sys.platform
    )
    return add_docstrings.transform(
        source,
        runtime_module,
        module_name=runtime_module.__name__,
        stub_file_path=Path(f"{runtime_module.__name__}.pyi"),
        typeshed_client_context=typeshed_client_context,
        blacklisted_objects=frozenset(),
    )


class DocumentedValue:
    """Runtime value whose instance docstring is useful for a stub attribute."""


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

        def __hidden(self) -> None:
            """Hidden method docs."""

    class D:
        """Class docs."""

    value = DocumentedValue()
    value.__doc__ = "Attribute docs."
    module.__dict__.update(f=f, g=g, C=C, D=D, x=value, y=value, z=value)
    type.__setattr__(C, "x", value)
    return module


def test_module_docstring_is_added() -> None:
    """The runtime module docstring is added before definition docstrings."""
    module = runtime_module()
    module.__doc__ = "Module docs."
    source = "def f(): ...\n"
    expected = textwrap.dedent(
        '''\
        """Module docs."""
        def f():
            """Function docs."""
        '''
    )
    assert transform(source, module) == expected


def test_class_docstring_is_added() -> None:
    """The runtime class docstring is added to the stub class."""
    source = "class D: ...\n"
    expected = textwrap.dedent(
        '''\
        class D:
            """Class docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


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


def test_name_mangled_method_is_resolved_from_runtime_class() -> None:
    """Private methods are looked up using their name-mangled runtime name."""
    source = textwrap.dedent(
        """\
        class C:
            def __hidden(self): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        class C:
            def __hidden(self):
                """Hidden method docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_logical_module_name_is_used_for_blacklist() -> None:
    """Blacklists use the stub's module name rather than the runtime alias."""
    module = runtime_module()
    source = "def f(): ...\n"
    context = typeshed_client.get_search_context(
        version=sys.version_info[:2], platform=sys.platform
    )
    transformed = add_docstrings.transform(
        source,
        module,
        module_name="logical_module",
        stub_file_path=Path("logical_module.pyi"),
        typeshed_client_context=context,
        blacklisted_objects=frozenset({"logical_module.f"}),
    )
    assert transformed == source


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


def test_non_ascii_prefix_does_not_shift_edits() -> None:
    """AST byte offsets do not corrupt character-based source edits."""
    # LibCST handles this correctly without special care, but a future AST-based
    # implementation could confuse UTF-8 byte offsets with string character offsets.
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
