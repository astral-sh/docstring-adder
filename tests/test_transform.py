from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import patch

import black
import pytest
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


def blacken(source: str) -> str:
    return black.format_str(source, mode=black.Mode(is_pyi=True))


def module_from_source(source: str) -> types.ModuleType:
    """Create a runtime module by executing the given source code."""
    module = types.ModuleType("test_module")
    exec(textwrap.dedent(source), module.__dict__)
    return module


def runtime_module() -> types.ModuleType:
    """Return a runtime module with documented definitions and attributes."""
    return module_from_source(
        '''\
        def f() -> None:
            """Function docs."""

        def g() -> None:
            """Other function docs."""

        class C:
            def method(self) -> None:
                """Method docs."""

            def __hidden(self) -> None:
                """Hidden method docs."""

        class Foo: ...

        x = Foo()
        x.__doc__ = "Attribute docs."

        y = z = x

        class D:
            """Class docs."""

            x = x
        '''
    )


def test_module_docstring_is_added() -> None:
    """The runtime module docstring is added before definition docstrings."""
    module = module_from_source(
        '''\
        """Module docs."""

        def f() -> None:
            """Function docs."""
        '''
    )
    source = "def f(): ...\n"
    expected = textwrap.dedent(
        '''\
        """Module docs."""
        def f():
            """Function docs."""
        '''
    )
    assert transform(source, module) == expected


def test_module_docstring_is_added_to_an_empty_stub() -> None:
    """A runtime module can document an otherwise empty marker stub."""
    module = module_from_source('"""Module docs."""')
    assert transform("", module) == '"""Module docs."""\n'


def test_existing_module_docstring_is_preserved() -> None:
    """A stub's existing module docstring is not replaced."""
    module = module_from_source('"""Runtime module docs."""')
    source = '"""Stub module docs."""\n'
    assert transform(source, module) == source


@pytest.mark.parametrize(
    ("runtime_docstring", "expected"),
    [
        (
            "A short docstring.",
            '''\
            def f() -> None:
                """A short docstring."""
            ''',
        ),
        (
            "A path containing a backslash: C:\\Users",
            '''\
            def f() -> None:
                """A path containing a backslash: C:\\\\Users"""
            ''',
        ),
        (
            """\
            A multiline docstring.

            More detail.""",
            '''\
            def f() -> None:
                """A multiline docstring.

                More detail.
                """
            ''',
        ),
        (
            "A docstring with a trailing newline.\n",
            '''\
            def f() -> None:
                """A docstring with a trailing newline."""
            ''',
        ),
        (
            "A docstring containing ''' triple quotes.",
            '''\
            def f() -> None:
                """A docstring containing \'\'\' triple quotes."""
            ''',
        ),
        (
            "Use \"\"\" or ''' for multiline strings.",
            """\
            def f() -> None:
                '''Use \"\"\" or \\'\\'\\' for multiline strings.'''
            """,
        ),
        (
            "",
            '''\
            def f() -> None:
                """"""
            ''',
        ),
    ],
)
def test_runtime_docstring_content_is_preserved_when_transforming(
    runtime_docstring: str, expected: str
) -> None:
    """Inserted function docstrings preserve realistic runtime content."""

    before = textwrap.dedent("""\
        def f() -> None: ...
        """)

    module = module_from_source(
        f"""\
        def f():
            {runtime_docstring!r}
        """
    )

    transformed = blacken(transform(before, module))
    assert transformed == textwrap.dedent(expected)


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        (
            """\
            def f():
                "A docstring containing ''' and ending with a double quote: \\""
            """,
            '''\
            def f():
                """A docstring containing \'\'\' and ending with a double quote: \\""""
            ''',
        ),
        (
            '''\
            def f():
                'A docstring containing """ and ending with a single quote: \\''
            ''',
            """\
            def f():
                '''A docstring containing \"\"\" and ending with a single quote: \\''''
            """,
        ),
    ],
)
def test_transform_escapes_a_final_docstring_delimiter_quote(
    runtime: str, expected: str
) -> None:
    """A final quote cannot merge with the selected triple-quote delimiter."""
    module = module_from_source(runtime)
    transformed = transform("def f(): ...\n", module)
    assert transformed == textwrap.dedent(expected)


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


def test_semicolon_delimited_statements_are_rejected() -> None:
    with pytest.raises(NotImplementedError, match="Semicolons are not supported"):
        transform("x: int; y: int\n", runtime_module())


def test_parenthesized_decorators_are_rejected() -> None:
    source = textwrap.dedent(
        """\
        class C:
            @(
                staticmethod
            )
            def method(): ...
        """
    )
    with pytest.raises(
        NotImplementedError, match="Parenthesized decorators are not supported"
    ):
        transform(source, runtime_module())


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


def test_decorated_first_method_in_class_is_documented() -> None:
    """A decorator is included when locating a suite's first statement."""
    source = textwrap.dedent(
        """\
        class C:
            @staticmethod
            def method(): ...
        """
    )
    expected = textwrap.dedent(
        '''\
        class C:
            @staticmethod
            def method():
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
    """A class absent at runtime is left unchanged."""
    source = textwrap.dedent(
        """\
        class Missing:
            __dict__: object
            class __class__: ...
            class __dict__: ...
        """
    )
    assert transform(source, runtime_module()) == source


def test_missing_runtime_function_is_unchanged() -> None:
    """An API absent from the runtime module is left alone."""
    source = "def optional_api() -> None: ...\n"
    assert transform(source, runtime_module()) == source


def test_functions_in_a_missing_runtime_class_are_unchanged() -> None:
    """Definitions nested under a missing runtime class remain untouched."""
    source = textwrap.dedent(
        """\
        class OptionalBackend:
            def connect(self) -> None: ...
        """
    )
    assert transform(source, runtime_module()) == source


def test_unreachable_missing_runtime_function_is_not_logged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An API absent because of a version guard does not produce a warning."""
    source = textwrap.dedent(
        """\
        import sys

        if sys.version_info >= (3, 15):
            def future_api() -> None: ...
        """
    )
    with patch.object(sys, "version_info", (3, 14)):
        assert transform(source, runtime_module()) == source
    assert "Could not find" not in capsys.readouterr().out


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


def test_existing_attribute_docstring_is_preserved() -> None:
    """An existing attribute docstring is not replaced."""
    source = textwrap.dedent(
        '''\
        x: int
        """Stub attribute docs."""
        '''
    )
    assert transform(source, runtime_module()) == source


def test_documented_assignment_gets_an_attribute_docstring() -> None:
    """A simple stub assignment can receive an attribute docstring."""
    source = "x = ...\n"
    expected = textwrap.dedent(
        '''\
        x = ...
        """Attribute docs."""
        '''
    )
    assert transform(source, runtime_module()) == expected


def test_blacklisted_attribute_is_unchanged() -> None:
    """Object blacklists apply to module attributes as well as definitions."""
    module = runtime_module()
    source = "x: int\n"
    context = typeshed_client.get_search_context(
        version=sys.version_info[:2], platform=sys.platform
    )
    transformed = add_docstrings.transform(
        source,
        module,
        module_name=module.__name__,
        stub_file_path=Path(f"{module.__name__}.pyi"),
        typeshed_client_context=context,
        blacklisted_objects=frozenset({f"{module.__name__}.x"}),
    )
    assert transformed == source


def test_type_alias_does_not_get_the_aliased_class_docstring() -> None:
    """A type alias is not documented with the runtime class's docstring."""
    module = module_from_source(
        '''\
        class D:
            """Class docs."""

        Alias = D
        '''
    )
    source = textwrap.dedent(
        """\
        from dependency import D

        Alias = D
        """
    )
    assert transform(source, module) == source


def test_annotated_type_alias_does_not_get_the_aliased_class_docstring() -> None:
    """An explicit TypeAlias is not documented with the aliased class's docs."""
    module = module_from_source(
        '''\
        class D:
            """Class docs."""

        Alias = D
        '''
    )
    source = textwrap.dedent(
        """\
        from dependency import D
        from typing_extensions import TypeAlias

        Alias: TypeAlias = D
        """
    )
    assert transform(source, module) == source


def test_callable_attribute_does_not_get_the_function_docstring() -> None:
    """A callable-valued variable is not documented as a function definition."""
    module = module_from_source(
        '''\
        def f() -> None:
            """Function docs."""

        factory = f
        '''
    )
    source = textwrap.dedent(
        """\
        from collections.abc import Callable

        factory: Callable[[], str]
        """
    )
    assert transform(source, module) == source


def test_undocumented_attribute_is_unchanged() -> None:
    """A runtime value without useful docs leaves its stub attribute unchanged."""
    module = module_from_source(
        """\
        class Undocumented:
            __doc__ = None

        undocumented = Undocumented()
        """
    )
    source = "undocumented: object\n"
    assert transform(source, module) == source


def test_generic_instance_docstring_is_not_added() -> None:
    """A generic runtime type's docstring is not copied to an instance attribute."""
    module = module_from_source("sentinel = object()")
    source = "sentinel: object\n"
    assert transform(source, module) == source


def test_inherited_object_method_docstring_is_not_added() -> None:
    """Generic documentation inherited from `builtins.object` is not copied into ordinary classes."""
    module = module_from_source(
        """\
        class Empty:
            pass
        """
    )
    source = textwrap.dedent(
        """\
        class Empty:
            def __init__(self) -> None: ...
        """
    )

    # `object.__init__.__doc__` is:
    #
    # > Initialize self.  See help(type(self)) for accurate signature.
    #
    # which isn't a useful docstring, so we don't add it to
    # `Empty.__init__` in the stub.
    assert transform(source, module) == source


def test_overridden_object_method_docstring_is_added() -> None:
    """Useful documentation on an overridden `object` method is preserved."""
    module = module_from_source(
        '''\
        class Custom:
            def __init__(self) -> None:
                """Custom initializer docs."""
        '''
    )
    source = textwrap.dedent(
        """\
        class Custom:
            def __init__(self) -> None: ...
        """
    )
    expected = textwrap.dedent(
        '''\
        class Custom:
            def __init__(self) -> None:
                """Custom initializer docs."""
        '''
    )
    assert transform(source, module) == expected


def test_named_tuple_field_docstrings_are_not_added() -> None:
    """Generated field-number docs are omitted from NamedTuple fields."""
    module = module_from_source(
        """\
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int
            y: int
        """
    )
    original_stub = textwrap.dedent(
        """\
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int
            y: int
        """
    )

    # We don't add the generated docstrings like "Alias for field number 0"
    # to the `x` and `y` attributes, since those aren't useful to users.
    expected_stub = textwrap.dedent(
        '''\
        from typing import NamedTuple

        class Point(NamedTuple):
            """Point(x, y)"""

            x: int
            y: int
        '''
    )

    transformed = blacken(transform(original_stub, module))

    assert transformed == expected_stub


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


def test_blank_line_is_inserted_before_if() -> None:
    """Nested `if` statements and their comments are separated from prior code."""
    source = textwrap.dedent(
        """\
        import sys
        if sys.version_info >= (3, 0):
            x: int
            # Version-specific APIs.
            if sys.version_info >= (3, 0):
                y: int
        """
    )
    expected = textwrap.dedent(
        """\
        import sys

        if sys.version_info >= (3, 0):
            x: int

            # Version-specific APIs.
            if sys.version_info >= (3, 0):
                y: int
        """
    )
    assert blacken(transform(source, types.ModuleType("test_module"))) == expected


@pytest.mark.parametrize(
    ("version_guard", "expected"),
    [
        (
            """\
            import sys

            if sys.version_info >= (3, 8):
                x: int

            def f() -> None: ...
            """,
            '''\
            import sys

            if sys.version_info >= (3, 8):
                x: int
                """Attribute docs."""

            def f() -> None:
                """Function docs."""
            ''',
        ),
        (
            """\
            import sys

            if sys.version_info >= (3, 15):
                y: int
            else:
                x: int

            def f() -> None: ...
            """,
            '''\
            import sys

            if sys.version_info >= (3, 15):
                y: int
            else:
                x: int
                """Attribute docs."""

            def f() -> None:
                """Function docs."""
            ''',
        ),
        (
            """\
            import sys

            if sys.version_info >= (3, 15):
                y: int
            elif sys.version_info >= (3, 8):
                x: int

            def f() -> None: ...
            """,
            '''\
            import sys

            if sys.version_info >= (3, 15):
                y: int
            elif sys.version_info >= (3, 8):
                x: int
                """Attribute docs."""

            def f() -> None:
                """Function docs."""
            ''',
        ),
    ],
)
def test_attribute_docstring_at_end_of_version_guard_is_followed_by_blank_line(
    version_guard: str, expected: str
) -> None:
    """A documented version-guard branch remains separated from following APIs."""
    source = textwrap.dedent(version_guard)
    expected = textwrap.dedent(expected)
    with patch.object(sys, "version_info", (3, 14)):
        transformed = blacken(transform(source, runtime_module()))
    assert transformed == expected


def test_multiline_attribute_docstring_in_else_if_is_indented() -> None:
    """A physical `else` suite contributes to attribute-docstring indentation."""
    module = module_from_source(
        """\
        class Foo: ...

        x = Foo()

        x.__doc__ = "First line.\\nSecond line."

        """
    )
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


def test_nonempty_source_without_trailing_newline() -> None:
    """A synthetic ENDMARKER row maps to the real end of the source."""
    source = "def f(): ..."
    expected = 'def f():\n    """Function docs."""'
    assert transform(source, runtime_module()) == expected


def test_non_ascii_prefix_does_not_shift_edits() -> None:
    """AST byte offsets do not corrupt character-based source edits."""
    module = module_from_source(
        '''\
        def π() -> None:
            """Function docs."""
        '''
    )
    source = "def π(): ...\n"
    expected = textwrap.dedent(
        '''\
        def π():
            """Function docs."""
        '''
    )
    assert transform(source, module) == expected


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
