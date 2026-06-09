"""Tool to add docstrings to stubs.

The tool is an AST-based codemod. Given a path to a typeshed `stdlib` directory or a
stubs package, docstring-adder will do the following:

1. Determine a set of all stub files in the directory (using the `typeshed_client`
   library).
2. For each stub file:
   a. We extract the module name of the stub file, again using the `typeshed_client`
      library.
   b. We attempt to import the corresponding runtime module. If the import fails, the
      failure is logged and the stub file is skipped; if not, we proceed to the next
      step.
   c. If the runtime module has a docstring, it is added to the stub file.
   d. For every class or function definition in the stub file:
      i. If the class/function definition already has a docstring, we skip it and move on
         to the next class/function definition. If not, we proceed to the next step.
      ii. We attempt to locate the corresponding runtime object by looking up the
         fullname of the class or function in the runtime module. If the runtime object
         can be found, we proceed to the next step; if not, a warning is logged and we
         skip to the next class or function definition.
      iii. If the runtime object has a docstring, the docstring is added to the
         class/function definition in the stub file.
   e: For every suite in the stub file (i.e., the body of a module, class, function,
      `if` block`, `elif` block, or `else` block), we also try to add attribute docstrings.
      These are rarer, because they are not usually preserved at runtime, but various
      descriptors can have docstrings that are accessible at runtime, which can be added
      to attribute or variable declarations in a stub file. Certain special forms in the
      typing module also benefit from this.
   f. The modified stub file is written back to disk.
   g. An AST safety check is performed to ensure that the modified stub file is still
      valid. It checks that the ASTs before and after docstring_adder's changes are
      identical, except for line numbers and added docstrings. If the modified stub file
      is not valid, an exception is raised. This is done after writing the modified stub
      file to disk so that it is possible to inspect the incorrect changes
      docstring-adder made.

Some miscellaneous details:
- The tool should be idempotent. It should add docstrings where possible, but it should
  never remove or alter docstrings that were already present in the stub file.
  Idempotency cannot be guaranteed, however, if importing the package at runtime causes
  persistent changes to be made to your Python environment.
- Because it applies targeted edits to the original source, it should not make
  spurious changes to formatting or comments in the stub file. `type: ignore` comments
  should be preserved; mypy and other type checkers should still be able to type-check
  the stub file after docstring-adder has run.
- Nested namespaces are supported: docstring-adder is capable of adding a docstring to a
  function definition inside a class (a method definition), a class definition inside a
  class, or even a function definition inside a class definition inside a class definition.
  Docstrings can even be added to name-mangled methods.
- docstring-adder skips adding a docstring to a method definition if the method docstring
  at runtime is exactly the same as the docstring of the corresponding method on `object`.
  This is to avoid adding a lot of boilerplate docstrings that are not useful.
- The tool should not add any docstrings to unreachable branches, given the platform and
  Python version it is run on. For example, if it is being run on Windows, it should not
  add any docstrings to definitions inside `if sys.platform == "linux"` branches;
  similarly, if it is being run on Python 3.10, it should not add any docstrings to
  definitions inside `if sys.version_info >= (3, 11)` branches. Fundamentally, the tool
  can only accurately add docstrings to definitions that exist at runtime on the Python
  version and platform the tool is run on, since docstrings are retrieved by dynamically
  inspecting the runtime module that corresponds to the stub.

  docstring-adder is not capable of type inference; whether or not these `if` tests
  evaluate to `True` is evaluated syntactically using APIs from `typeshed_client`.
- For an overloaded function, docstring-adder will only add a docstring to the first
  overload in any given suite. For example:

  ```py
  import sys
  from typing import overload

  if sys.platform == "linux":

      @overload
      def foo(x: int) -> int:
          '''Linux-specific docs.'''

      @overload
      def foo(x: str) -> str: ...

  else:

      @overload
      def foo(x: int) -> str:
          '''Docs for foo on other platforms.'''

      @overload
      def foo(x: str) -> int: ...
  ```

"""

from __future__ import annotations

import argparse
import ast
import bisect
import contextlib
import importlib
import inspect
import io
import subprocess
import sys
import textwrap
import token
import tokenize
import types
import typing
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from operator import attrgetter
from pathlib import Path
from typing import Any, ClassVar, NewType, TypeAlias, TypeGuard

import tomli
import typeshed_client
from rich_argparse import RawDescriptionRichHelpFormatter
from termcolor import colored


def log(*objects: object) -> None:
    """Log a warning to the terminal."""
    print(colored(" ".join(map(str, objects)), "yellow"))


# Type alias representing a node that could have a docstring added to it.
Documentable: TypeAlias = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
# NewTypes representing distinct integer domains used for source locations.
TokenKind = NewType("TokenKind", int)
LineNumber = NewType("LineNumber", int)
CharacterColumn = NewType("CharacterColumn", int)
ByteColumn = NewType("ByteColumn", int)
TokenListIndex = NewType("TokenListIndex", int)
TextOffset = NewType("TextOffset", int)


@dataclass(frozen=True, slots=True)
class Token:
    """A token with absolute character offsets into the original source."""

    kind: TokenKind
    string: str
    start_line: LineNumber
    index: TokenListIndex
    startpos: TextOffset
    endpos: TextOffset


class TokenIndex:
    """Map AST source coordinates to stdlib tokens in the original source."""

    _NON_CODING_TOKENS: ClassVar[set[TokenKind]] = {
        TokenKind(token.INDENT),
        TokenKind(token.DEDENT),
        TokenKind(token.NEWLINE),
        TokenKind(token.ENDMARKER),
        TokenKind(token.COMMENT),
        TokenKind(tokenize.NL),
    }

    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.split("\n")
        self.line_starts = [TextOffset(0)]
        self.line_starts.extend(
            TextOffset(index + 1)
            for index, character in enumerate(source)
            if character == "\n"
        )

        token_infos = tokenize.generate_tokens(io.StringIO(source).readline)
        self.tokens: list[Token] = []
        for index, token_info in enumerate(token_infos):
            start_line = LineNumber(token_info.start[0])
            start_column = CharacterColumn(token_info.start[1])
            end_line = LineNumber(token_info.end[0])
            end_column = CharacterColumn(token_info.end[1])
            self.tokens.append(
                Token(
                    kind=TokenKind(token_info.type),
                    string=token_info.string,
                    start_line=start_line,
                    index=TokenListIndex(index),
                    startpos=self.source_offset(start_line, start_column),
                    endpos=self.source_offset(end_line, end_column),
                )
            )
        self.token_starts = [source_token.startpos for source_token in self.tokens]

    def source_offset(
        self, line: LineNumber, character_column: CharacterColumn
    ) -> TextOffset:
        """Convert a tokenizer row and character column to a source offset."""

        if line > len(self.line_starts):
            return TextOffset(len(self.source))
        return TextOffset(self.line_starts[line - 1] + character_column)

    def ast_offset(self, line: LineNumber, byte_column: ByteColumn) -> TextOffset:
        """Convert an AST row and UTF-8 byte column to a source offset."""

        source_line = self.lines[line - 1]
        character_column = len(
            source_line.encode("utf-8")[:byte_column].decode("utf-8")
        )
        return TextOffset(self.line_starts[line - 1] + character_column)

    def first_token(self, node: ast.stmt) -> Token:
        """Return the first coding token belonging to a statement."""

        first_node = (
            node.decorator_list[0]
            if isinstance(node, Documentable) and node.decorator_list
            else node
        )
        start = self.ast_offset(
            LineNumber(first_node.lineno), ByteColumn(first_node.col_offset)
        )
        first_index = bisect.bisect_left(self.token_starts, start)

        if (
            first_node is not node
            and first_index > 0
            and self.tokens[first_index - 1].string == "@"
        ):
            first_index -= 1

        while self.tokens[first_index].kind in self._NON_CODING_TOKENS:
            first_index += 1
        return self.tokens[first_index]

    def last_token(self, node: ast.stmt) -> Token:
        """Return the last coding token belonging to a statement."""

        end_line = node.end_lineno
        end_column = node.end_col_offset
        if end_line is None or end_column is None:
            raise RuntimeError("AST statement is missing end position information")

        end = self.ast_offset(LineNumber(end_line), ByteColumn(end_column))
        last_index = bisect.bisect_right(self.token_starts, end) - 1
        while (
            self.tokens[last_index].endpos > end
            or self.tokens[last_index].kind in self._NON_CODING_TOKENS
        ):
            last_index -= 1
        return self.tokens[last_index]


def _is_statement_list(value: list[Any]) -> TypeGuard[list[ast.stmt]]:
    return all(isinstance(item, ast.stmt) for item in value)


def _is_string_statement(node: ast.stmt) -> bool:
    match node:
        case ast.Expr(value=ast.Constant(value=str())):
            return True
        case _:
            return False


def _real_elif_of_if(node: ast.If, token_index: TokenIndex) -> ast.If | None:
    """If the AST's nested `If` was written as an `elif` clause, return that nested `If`.

    Returns `None` if this `If` does not have an `elif` clause.

    The stdlib AST represents `elif condition:` and `else: if condition:` in the
    same structural shape, so the source token is required to distinguish them.
    """
    match node.orelse:
        case [ast.If() as elif_node]:
            if token_index.first_token(elif_node).string == "elif":
                return elif_node
    return None


def _final_statement_of_if(node: ast.If) -> ast.stmt:
    """Retrieve the final statement of an `if`/`elif`/`else` chain."""
    match node.orelse:
        case []:
            return node.body[-1]
        case [ast.If() as nested_if]:
            return _final_statement_of_if(nested_if)
        case _:
            return node.orelse[-1]


def _child_suites(node: ast.AST) -> list[list[ast.stmt]]:
    """Return direct child suites contained in a non-namespace AST node.

    Some suites are nested inside helper nodes such as `ExceptHandler` and
    `match_case`, so this descends through non-expression, non-statement container
    nodes without recursively visiting the statements themselves.
    """
    suites: list[list[ast.stmt]] = []
    for _field_name, value in ast.iter_fields(node):
        if not isinstance(value, list) or not value:
            continue
        if _is_statement_list(value):
            suites.append(value)
        else:
            for item in value:
                if isinstance(item, ast.AST) and not isinstance(
                    item, (ast.expr, ast.stmt)
                ):
                    suites.extend(_child_suites(item))
    return suites


def triple_quoted_docstring(content: str, indentation: str | None = None) -> str:
    """Escape the docstring and return it as a triple-quoted string.

    Logic adapted from `ast.unparse()` internals.
    See https://github.com/python/cpython/blob/9a6b60af409d02468b935c569a4f49e88c399c4e/Lib/_ast_unparse.py#L532-L568
    """

    def escape_char(c: str) -> str:
        # \n and \t are non-printable but we wouldn't want them to be escaped.
        if c.isspace():
            return c
        # Always escape backslashes and other non-printable characters
        if c == "\\" or not c.isprintable():
            return c.encode("unicode_escape").decode("ascii")
        return c

    # In general we try to tamper with the content as little as possible,
    # but this generally leads to more consistent and more readable docstrings.
    newline_count = content.count("\n")
    if newline_count > 0:
        ends_with_newline = content.rstrip(" \t").endswith("\n")
        if ends_with_newline:
            if (
                newline_count == 1
                and content.strip()
                and content.rstrip(" \t")[-2] not in {'"', "'"}
            ):
                content = content.rstrip(" \t").removesuffix("\n")
        else:
            content = content.rstrip(" \t") + "\n"
            if indentation is not None:
                content += indentation

    escaped_string = "".join(map(escape_char, content))

    # Why would somebody add an empty docstring...??
    # No idea. But, empirically, it happens.
    if not escaped_string:
        return '"' * 6

    quotes = ['"""', "'''"]
    possible_quotes = [q for q in quotes if q not in escaped_string]

    if not possible_quotes:
        string = repr(content)
        quote = next((q for q in quotes if string[0] in q), string[0])
        return f"{quote}{string[1:-1]}{quote}"

    # Sort so that we prefer '''"''' over """\""""
    possible_quotes.sort(key=lambda q: q[0] == escaped_string[-1])
    quote = possible_quotes[0]

    # Escape the final quote, if necessary
    if quote[0] == escaped_string[-1]:
        escaped_string = escaped_string[:-1] + "\\" + escaped_string[-1]

    return f"{quote}{escaped_string}{quote}"


@dataclass
class RuntimeValue:
    """An arbitrary value at runtime (that may or may not have a docstring).

    We use a wrapper class here for stricter typing:
    the runtime object corresponding to a class/function in a stub file
    could be literally anything, but using `object` or `Any` would
    make the code difficult to confidently refactor: type checkers would
    allow instances of *any* type to be passed around, meaning they often
    wouldn't spot mistakes in the code.
    """

    inner: object

    def is_not_found(self) -> bool:
        """Return `True` if the runtime object could not be found."""
        return self.inner is NOT_FOUND


@dataclass
class RuntimeParent:
    """Information regarding the namespace `DocstringAdder` is currently visiting."""

    __slots__ = {
        "name": """The (unqualified) name of the current namespace:

                For example, if we're visiting a class definition `Bar` inside a class
                definition `Baz` inside a module `spam`, the name will be `Bar` (*not*
                `spam.Baz.Bar`).
                """,
        "value": """The runtime value of the namespace.

                Usually this will be an instance of `type` (a class object) or an
                instance of `types.ModuleType` (a module object). It could theoretically
                be anything, however; don't make any assumptions about it!
                """,
    }

    name: str
    value: RuntimeValue


@dataclass(frozen=True, kw_only=True, slots=True)
class SourceEdit:
    """Replacement text for a half-open range.

    An empty range signifies an insertion.
    """

    start: TextOffset
    end: TextOffset
    text: str


class SourceEditor:
    """Collect non-overlapping edits and apply them without reformatting the source."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.edits: list[SourceEdit] = []

    def replace(self, start: TextOffset, end: TextOffset, replacement: str) -> None:
        """Replace the half-open source range from start to end."""
        self.edits.append(SourceEdit(start=start, end=end, text=replacement))

    def insert(self, offset: TextOffset, text: str) -> None:
        """Insert text, preserving call order at a shared text offset."""
        self.edits.append(SourceEdit(start=offset, end=offset, text=text))

    def render(self) -> str:
        """Apply the planned source edits."""

        edits = sorted(self.edits, key=attrgetter("start", "end"))
        rendered: list[str] = []
        source_position = TextOffset(0)
        previous_edit: SourceEdit | None = None
        for edit in edits:
            if edit.start < source_position:
                raise RuntimeError(
                    "docstring-adder planned conflicting source edits; "
                    f"{previous_edit!r} overlaps {edit!r}. "
                    "This indicates a bug in the source editor."
                )
            rendered.extend((self.source[source_position : edit.start], edit.text))
            source_position = edit.end
            previous_edit = edit

        rendered.append(self.source[source_position:])
        return "".join(rendered)


class DocstringAdder:
    """Visitor to add docstrings to a stub file.

    The visitor recursively attempts to add docstrings to all reachable class and
    function definitions inside a parsed stub file. The stdlib AST provides the
    structure used for traversal, while stdlib tokenization locates the exact source
    boundaries needed to preserve the file's existing formatting and comments.

    Edits are collected during traversal and applied afterwards. Applying an edit
    immediately would invalidate the token offsets used to plan subsequent edits.
    """

    def __init__(
        self,
        *,
        source: str,
        parsed_module: ast.Module,
        module_name: str,
        runtime_module: types.ModuleType,
        stub_file_path: Path,
        typeshed_client_context: typeshed_client.SearchContext,
        blacklisted_objects: frozenset[str],
    ) -> None:
        # All edit offsets are character positions in the original source.
        self.source = source
        self.token_index = TokenIndex(source)
        self.parsed_module = parsed_module

        # Source edits are accumulated here and applied after the traversal finishes.
        self.editor = SourceEditor(source)

        # A stack containing the runtime namespaces corresponding to the AST namespace
        # currently being visited. Class definitions push an entry; function
        # definitions deliberately do not, matching their runtime lookup semantics.
        self.runtime_parents: list[RuntimeParent] = [
            RuntimeParent(name=module_name, value=RuntimeValue(inner=runtime_module))
        ]
        self.stub_file_path = stub_file_path
        self.typeshed_client_context = typeshed_client_context
        self.blacklisted_objects = blacklisted_objects

        # A stack of sets. Each set corresponds to a physical indented suite in the
        # source. For each nested suite, we keep track of names of functions that we
        # have already visited (and, possibly, added docstrings to).
        #
        # We start with a single empty set corresponding to the module-level namespace.
        # The module body is not an indented suite, but for our purposes it is treated
        # in the same way.
        self.suite_visitation_stack: list[set[str]] = [set()]

        # Planned source edits are not visible in the AST. Keep track of assignments
        # receiving attribute docstrings so enclosing `if` statements can still apply
        # the correct blank-line rule.
        self.attribute_documented: set[ast.stmt] = set()

        # When an inline suite such as `def f(): ...` is expanded, use the first
        # indentation token in the file, falling back to four spaces for flat files.
        self.default_indent = next(
            (
                source_token.string
                for source_token in self.token_index.tokens
                if source_token.kind == TokenKind(token.INDENT)
            ),
            "    ",
        )

        # A NEWLINE token has an empty string at the end of the file. This fallback is
        # used when a separator is needed to insert a new statement while preserving
        # whether the original file ended with a newline.
        self.default_newline = "\r\n" if "\r\n" in source else "\n"

    def transform(self) -> str:
        """Plan all docstring edits and return the transformed source."""

        for statement in self.parsed_module.body:
            self.visit_statement(statement, reachable=True)
        self._plan_attribute_docstrings(self.parsed_module.body, indentation=0)
        return self.editor.render()

    def maybe_mangled_name(self, name: str) -> str:
        """Apply name mangling to `name`, if it is necessary.

        Given the name of a class or function that would be implied by naively
        reading a stub's source code, this function returns the name that the
        given class or function actually has at runtime. This will usually be
        the same as `name`, but if the source name starts with `__` and does
        not end with `__` and is defined inside a class namespace, name
        mangling will be applied at runtime.

        For example, naively you would expect the name of the `__method`
        method here to be `__method`, but in fact at runtime it needs to be
        looked up as `_Foo__method` on the `Foo` class:

            >>> class Foo:
            ...     def __method(self): ...
            >>> Foo.__method
            Traceback (most recent call last):
            File "<python-input-1>", line 1, in <module>
                Foo.__method
            AttributeError: type object 'Foo' has no attribute '__method'
            >>> Foo._Foo__method
            <function Foo.__method at 0x105a1e020>
        """
        parent = self.runtime_parents[-1]
        if not isinstance(parent.value.inner, type):
            return name

        if name.startswith("__") and not name.endswith("__"):
            return f"_{parent.name.lstrip('_')}{name}"

        return name

    def object_fullname(self, final_part: str) -> str:
        """Given an unqualified name, convert it to a fully qualified name.

        For example, given the name `method`, if we are visiting a class `Foo`
        inside a class `Bar` inside a module `spam`, this will return `spam.Bar.Foo.method`.

        If the name starts with `__`, it will be mangled according to the current class
        namespace: if we were given the name `__method`, we would instead return
        `spam.Bar.Foo._Foo__method`.
        """
        parent_fullname = ".".join(parent.name for parent in self.runtime_parents)
        mangled_name = self.maybe_mangled_name(final_part)
        return f"{parent_fullname}.{mangled_name}"

    def _log_runtime_object_not_found(self, name: str, *, reachable: bool) -> None:
        if reachable:
            log(f"Could not find {self.object_fullname(name)} at runtime")

    def visit_statement(self, node: ast.stmt, *, reachable: bool) -> None:
        """Visit a statement and recursively visit any suites it contains."""

        match node:
            case ast.ClassDef():
                self.visit_class(node, reachable=reachable)
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                self.visit_function(node, reachable=reachable)
            case ast.If():
                self.visit_if(node, reachable=reachable)
            case _:
                for child_suite in _child_suites(node):
                    self.visit_suite(child_suite, reachable=reachable)

    def visit_class(self, node: ast.ClassDef, *, reachable: bool) -> None:
        """Visit a class definition and attempt to add its runtime docstring.

        The runtime object representing the class is retrieved before the class body is
        visited and appended to the stack of runtime parents. This allows nested classes,
        methods, and attributes to be resolved relative to the runtime class. The class
        docstring itself is planned after all statements in the class body have been
        visited.
        """
        runtime_parent = self.runtime_parents[-1]
        runtime_object: RuntimeValue | None
        if runtime_parent.value.is_not_found():
            runtime_object = RuntimeValue(NOT_FOUND)
        else:
            runtime_object = get_runtime_object_for_stub(
                name=self.maybe_mangled_name(node.name), runtime_parent=runtime_parent
            )
            if runtime_object is None:
                self._log_runtime_object_not_found(node.name, reachable=reachable)
                runtime_object = RuntimeValue(NOT_FOUND)
        self.runtime_parents.append(RuntimeParent(node.name, runtime_object))

        self.visit_suite(node.body, reachable=reachable)

        runtime_class = self.runtime_parents.pop().value
        if not runtime_class.is_not_found():
            self._document_class_or_function(
                node, runtime_object=runtime_class, reachable=reachable
            )

    def visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, reachable: bool
    ) -> None:
        """Visit a function definition and attempt to add its runtime docstring."""

        self.visit_suite(node.body, reachable=reachable)

        # If there are multiple functions with the same name in an indented block,
        # it's probably an overloaded function or the `@setter` for a property.
        # Only add a docstring for the first definition.
        if node.name in self.suite_visitation_stack[-1]:
            return
        self.suite_visitation_stack[-1].add(node.name)

        runtime_parent = self.runtime_parents[-1]

        if runtime_parent.value.is_not_found():
            return

        runtime_object = get_runtime_object_for_stub(
            name=self.maybe_mangled_name(node.name), runtime_parent=runtime_parent
        )

        if runtime_object is None:
            self._log_runtime_object_not_found(node.name, reachable=reachable)
            return

        self._document_class_or_function(
            node, runtime_object=runtime_object, reachable=reachable
        )

    def visit_if(self, node: ast.If, *, reachable: bool) -> None:
        """Visit an `if` statement and only document its reachable branch.

        The test is evaluated syntactically using `typeshed_client`.
        """
        condition = typeshed_client.evaluate_expression_truthiness(
            node.test, ctx=self.typeshed_client_context, file_path=self.stub_file_path
        )
        assert isinstance(condition, bool)

        self.visit_suite(node.body, reachable=reachable and condition)

        maybe_elif = _real_elif_of_if(node, self.token_index)

        if maybe_elif is not None:
            self.visit_if(maybe_elif, reachable=reachable and not condition)
        elif node.orelse:
            self.visit_suite(node.orelse, reachable=reachable and not condition)

    def visit_suite(self, body: list[ast.stmt], *, reachable: bool) -> None:
        """Visit the statements in a suite and plan its attribute docstrings.

        The stdlib AST represents both inline suites and physical indented suites as
        statement lists. Only physical indented suites receive their own function-name
        set and attribute-docstring pass; inline suites do not.
        """
        is_indented = self._is_indented_suite(body)
        if is_indented:
            self.suite_visitation_stack.append(set())

        for statement in body:
            self.visit_statement(statement, reachable=reachable)

        if is_indented:
            # Each non-module stack entry is one physical indentation level. This is
            # used to format multiline attribute docstrings at four spaces per level.
            indentation = len(self.suite_visitation_stack) - 1
            self.suite_visitation_stack.pop()

            # Don't add attribute docstrings to fields in `NamedTuple` classes.
            # The generated docstrings for the properties in these classes
            # are always things like "Alias for field number 0", which clutter
            # the stubs and aren't useful.
            if (
                reachable
                and not self.runtime_parents[-1].value.is_not_found()
                and not self._runtime_parent_is_named_tuple()
            ):
                self._plan_attribute_docstrings(body, indentation=indentation)

    def _document_class_or_function(
        self, node: Documentable, *, runtime_object: RuntimeValue, reachable: bool
    ) -> None:
        """Attempt to add a docstring to a class or function definition."""

        if not reachable:
            return

        object_fullname = self.object_fullname(node.name)
        if object_fullname in self.blacklisted_objects:
            return

        if ast.get_docstring(node, clean=False) is not None:
            return

        docstring = get_runtime_docstring(runtime=runtime_object)
        if docstring is None:
            return

        # E.g. we want to avoid a bajillion `__init__` docstrings that are just
        #
        # > Initialize self.  See help(type(self)) for accurate signature.
        #
        # Which is exactly what we get for `help(object.__init__)`
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not isinstance(runtime_object.inner, types.ModuleType)
            and node.name in object.__dict__
        ):
            method_docstring_on_object = get_runtime_docstring(
                runtime=RuntimeValue(inner=object.__dict__[node.name])
            )
            if docstring == method_docstring_on_object:
                return

        self._plan_definition_docstring(node, docstring)

    def _plan_definition_docstring(self, node: Documentable, docstring: str) -> None:
        """Plan the source edit that inserts a class or function docstring."""

        first_body_token = self.token_index.first_token(node.body[0])
        suite_colon = self._previous_token_with_string(first_body_token, ":")
        header_newline = self._next_token_of_kind(suite_colon, TokenKind(token.NEWLINE))
        literal = triple_quoted_docstring(docstring)

        if first_body_token.start_line == suite_colon.start_line:
            # If the body is just a `...`, replace it with just the docstring.
            match node.body:
                case [ast.Expr(value=ast.Constant(value=types.EllipsisType()))]:
                    pass
                case _:
                    return

            # but preserve `type: ignore` comments
            trailing_comment = self._trailing_comment(
                self.token_index.last_token(node.body[0]), header_newline
            )
            newline = self._newline_text(header_newline)
            child_indent = (
                self._line_indentation(suite_colon.startpos) + self.default_indent
            )
            replacement = (
                f"{trailing_comment}{newline}{child_indent}{literal}"
                f"{header_newline.string}"
            )
            self.editor.replace(suite_colon.endpos, header_newline.endpos, replacement)
            return

        body_indent = self._line_indentation(first_body_token.startpos)
        # For a physical indented suite, insert the docstring before the first body
        # statement. A multiline `...` body is preserved rather than replaced.
        newline = self._newline_text(header_newline)
        prefix = "" if header_newline.string else newline
        self.editor.insert(
            header_newline.endpos, f"{prefix}{body_indent}{literal}{newline}"
        )

    def _plan_attribute_docstrings(
        self, body: list[ast.stmt], *, indentation: int
    ) -> None:
        """Add Sphinx-style 'attribute docstrings' to assignments in a suite.

        The suite could be the body of a module, class, function,
        `if` block, `elif` block, or `else` block.
        """
        # Semicolon-separated statements share a terminal NEWLINE token. Count all
        # statements by that token's ending character offset so attribute docstrings are
        # only added to assignments that are the sole statement on their physical line.
        line_counts = Counter(
            self._terminal_newline(statement).endpos for statement in body
        )
        runtime_parent = self.runtime_parents[-1]

        for index, statement in enumerate(body):
            next_statement = body[index + 1] if index + 1 < len(body) else None

            if isinstance(statement, ast.If):
                final_statement = _final_statement_of_if(statement)
                if (
                    _is_string_statement(final_statement)
                    or final_statement in self.attribute_documented
                ):
                    self._ensure_blank_line_before_next(final_statement, next_statement)
                    continue

            # If it's an assignment that we could potentially add a docstring to,
            # and it is the only statement on its physical line...
            statement_newline = self._terminal_newline(statement)
            if line_counts[statement_newline.endpos] != 1:
                continue
            # ... and there is no docstring already present...
            if next_statement is not None and _is_string_statement(next_statement):
                continue

            assignment: ast.Assign | ast.AnnAssign
            target: ast.expr
            match statement:
                case ast.AnnAssign(target=target):
                    assignment = statement
                case ast.Assign(targets=[target]):
                    assignment = statement
                case _:
                    continue

            if not isinstance(target, ast.Name):
                continue

            # ... then try to add a docstring to it.
            runtime_name = target.id
            runtime_fullname = f"{runtime_parent.name}.{runtime_name}"
            if runtime_fullname in self.blacklisted_objects:
                continue

            runtime_value = get_runtime_object_for_stub(runtime_name, runtime_parent)

            if runtime_value is None:
                continue

            # -------------------------------------------------------------------------
            # BEGINNING of heuristics to avoid adding undesirable attribute docstrings.
            # -------------------------------------------------------------------------

            # Don't add the module docstring of `some_module` below `x = some_module`
            if isinstance(runtime_value.inner, types.ModuleType):
                continue

            if isinstance(
                runtime_value.inner,
                (
                    type,
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    types.GenericAlias,
                    types.MethodWrapperType,
                ),
            ):
                # Don't add the function docstring below `x = some_function`
                if isinstance(assignment, ast.Assign):
                    continue

                # Don't add the docstring for `dict` below `x: TypeAlias = dict[str, Any]`
                if assignment.value is not None:
                    continue

                # Don't add docstrings to things that look like `x: type[Foo]`
                # if the runtime value is a class.
                # Also, don't add docstrings to things that look like `x: Callable[..., Foo]`
                # if the runtime value is a function.
                #
                # ... But exclude `typing.Generic` here.
                # It's annotated with `Generic: type[_Generic]` in typeshed,
                # at least currently, and that's an *extremely* useful docstring :-(
                if (
                    isinstance(runtime_value.inner, (type, types.FunctionType))
                    and isinstance(assignment.annotation, ast.Subscript)
                    and runtime_fullname != "typing.Generic"
                ):
                    continue

                # Don't add the docstring for a function or class below `x: Foo`
                # if the runtime function or class comes from another module
                try:
                    runtime_module = runtime_value.inner.__module__
                except Exception:
                    pass
                else:
                    if isinstance(runtime_parent.value.inner, types.ModuleType):
                        parent_module = runtime_parent.value.inner.__name__
                    else:
                        try:
                            parent_module = runtime_parent.value.inner.__module__
                        except Exception:
                            parent_module = None
                    if parent_module is not None and runtime_module != parent_module:
                        continue

            # --------------------------------------------------------------------
            # END heuristics for avoiding adding undesirable attribute docstrings.
            # --------------------------------------------------------------------

            docstring = get_runtime_docstring(runtime_value)
            if docstring is None:
                continue

            docstring = docstring.strip(" \t")
            docstring_lines = docstring.split("\n")
            docstring = "\n".join([
                docstring_lines[0],
                textwrap.dedent("\n".join(docstring_lines[1:])),
            ])

            # If we're visiting an indented block, indent the docstring
            if indentation and "\n" in docstring:
                indentation_string = "    " * indentation
                docstring = (
                    textwrap.indent(docstring, prefix=indentation_string)
                    + indentation_string
                ).lstrip(" \t")
            else:
                indentation_string = None

            source_indent = self._line_indentation(
                self.token_index.first_token(statement).startpos
            )
            newline = self._newline_text(statement_newline)
            prefix = "" if statement_newline.string else newline
            literal = triple_quoted_docstring(docstring, indentation=indentation_string)
            self.editor.insert(
                statement_newline.endpos,
                f"{prefix}{source_indent}{literal}{statement_newline.string}",
            )
            self.attribute_documented.add(statement)
            self._ensure_blank_line_before_next(statement, next_statement)

    def _ensure_blank_line_before_next(
        self, statement: ast.stmt, next_statement: ast.stmt | None
    ) -> None:
        """Plan a blank line after an added docstring when one is not already present."""

        if next_statement is None:
            return

        # If we just added a docstring to the previous statement,
        # add a blank line before this statement.
        # Black will not do this for us.
        statement_newline = self._terminal_newline(statement)
        next_statement_start = self.token_index.first_token(next_statement).startpos
        next_line_start = self._line_start(next_statement_start)
        gap = self.source[statement_newline.endpos : next_line_start]
        if all(line.strip() for line in gap.splitlines()):
            blank_line_indent = self._line_indentation(next_statement_start)
            self.editor.insert(
                statement_newline.endpos,
                blank_line_indent + self._newline_text(statement_newline),
            )

    def _runtime_parent_is_named_tuple(self) -> bool:
        """Return whether the current runtime parent is a NamedTuple-like class."""

        runtime_parent = self.runtime_parents[-1].value.inner
        return (
            isinstance(runtime_parent, type)
            and issubclass(runtime_parent, tuple)
            and hasattr(runtime_parent, "_fields")
        )

    def _is_indented_suite(self, body: list[ast.stmt]) -> bool:
        first_token = self.token_index.first_token(body[0])
        suite_colon = self._previous_token_with_string(first_token, ":")
        return suite_colon.start_line != first_token.start_line

    def _previous_token_with_string(self, start: Token, string: str) -> Token:
        for index in range(start.index - 1, -1, -1):
            candidate = self.token_index.tokens[index]
            if candidate.string == string:
                return candidate
        raise RuntimeError(f"Could not find preceding {string!r} token")

    def _next_token_of_kind(self, start: Token, token_kind: TokenKind) -> Token:
        for candidate in self.token_index.tokens[start.index :]:
            if candidate.kind == token_kind:
                return candidate
        raise RuntimeError(f"Could not find token kind {token_kind}")

    def _terminal_newline(self, node: ast.stmt) -> Token:
        return self._next_token_of_kind(
            self.token_index.last_token(node), TokenKind(token.NEWLINE)
        )

    def _trailing_comment(self, last_body_token: Token, newline: Token) -> str:
        saw_semicolon = False
        for candidate in self.token_index.tokens[
            last_body_token.index + 1 : newline.index
        ]:
            if candidate.string == ";":
                saw_semicolon = True
            if candidate.kind == TokenKind(token.COMMENT):
                if saw_semicolon:
                    return self.source[candidate.startpos : candidate.endpos]
                whitespace_start = candidate.startpos
                while (
                    whitespace_start > last_body_token.endpos
                    and self.source[whitespace_start - 1] in " \t"
                ):
                    whitespace_start = TextOffset(whitespace_start - 1)
                return self.source[whitespace_start : candidate.endpos]
        return ""

    def _newline_text(self, newline: Token) -> str:
        return newline.string or self.default_newline

    def _line_start(self, position: TextOffset) -> TextOffset:
        return TextOffset(self.source.rfind("\n", 0, position) + 1)

    def _line_indentation(self, position: TextOffset) -> str:
        line_start = self._line_start(position)
        indentation_end = line_start
        while (
            indentation_end < len(self.source)
            and self.source[indentation_end] in " \t\f"
        ):
            indentation_end = TextOffset(indentation_end + 1)
        return self.source[line_start:indentation_end]


def get_runtime_docstring(runtime: RuntimeValue) -> str | None:
    """Attempt to retrieve the docstring for a given runtime object.

    We try to "tamper" with the docstring as little as possible after
    retrieving it. For example, we don't use `inspect.cleandoc()` here:
    the changes it makes are mostly unnecessary given that we apply
    autoformatting to stub files after docstring-adder has run, and it
    sometimes makes spurious/undesirable changes to the docstring.
    """
    runtime_object = runtime.inner

    try:
        # Don't use `inspect.getdoc()` here.
        #
        # If you call `inspect.getdoc(Foo.bar)`, where `Foo.bar` does not have a docstring,
        # but `Foo` inherits from `Spam` and `Spam.bar` *does* have a docstring,
        # `inspect.getdoc()` will return the docstring from `Spam.bar`.
        #
        # That's not what we want here: if the overridden method does not have a docstring in
        # the source code at runtime, we shouldn't add one in the stub file either.
        runtime_docstring = runtime_object.__doc__
    except Exception:
        return None

    if not isinstance(runtime_docstring, str):
        return None

    # These assertions are regression tests against bugs that were in early versions
    # of docstring-adder, where the docstring for `staticmethod` itself would be
    # added to the stub definition of a staticmethod.
    # This was fixed by using `inspect.unwrap()` in `get_runtime_object_for_stub()`.
    if runtime_object is not staticmethod:
        assert runtime_docstring != staticmethod.__doc__, runtime
    if runtime_object is not classmethod:
        assert runtime_docstring != classmethod.__doc__, runtime
    if runtime_object is not property:
        assert runtime_docstring != property.__doc__, runtime

    # For example, `TYPE_CHECKING` is just a `bool`, so
    # its docstring will just be the same as `bool.__doc__`,
    # which doesn't actually provide any information about the
    # `TYPE_CHECKING` variable itself. But we *do* want to add
    # attribute docstrings for the various `sys`-module variables
    # such as `sys.version_info`, `sys.float_info`, etc.,
    # even though these variables have the same docstrings as their
    # classes.
    if (
        runtime_object is not type
        and type(runtime_object).__module__ != "sys"
        and (
            runtime_docstring
            == get_runtime_docstring(RuntimeValue(type(runtime_object)))
        )
    ):
        return None

    return runtime_docstring


class NOT_FOUND:
    """Sentinel to indicate the runtime object for a stub definition could not be found.

    A custom sentinel is required because `None` might be the corresponding runtime object
    for many stub definitions.
    """


def get_runtime_object_for_stub(
    name: str, runtime_parent: RuntimeParent
) -> RuntimeValue | None:
    """Retrieve the runtime object corresponding to a stub definition.

    Specifically, given the name of an object at runtime and the parent namespace
    of that object at runtime, this function attempts to retrieve the runtime object
    from that namespace.

    For some edge cases (special `sys`-module APIs, `typing`-module aliases to objects
    in `collections.abc`, etc.), we may return a *slightly* different object than what
    would be implied directly by the name passed in, if it will result in the tool
    being able to add strictly superior docstrings to the stub definition.
    """

    # Typeshed reports that the type of `sys.float_info` is a class called `sys._float_info`,
    # but no such class exists at runtime. Pragmatically, it's better here if we return the
    # runtime object for `type(sys.float_info)` here. Although `sys.float_info` itself has
    # an attribute docstring added to it elsewhere, returning the runtime object for
    # `type(sys.float_info)` here has the added advantage that we recurse into methods and
    # properties defined on the class as well.
    if runtime_parent.value.inner is sys and name in {
        "_float_info",
        "_flags",
        "_int_info",
        "_hash_info",
        "_thread_info",
        "_version_info",
    }:
        name = name[1:]

    try:
        runtime = inspect.unwrap(
            inspect.getattr_static(runtime_parent.value.inner, name)
        )

    # Some getattr() and `hasattr()` calls raise TypeError,
    # or something even more exotic,
    # but we don't want the tool to crash in these cases.
    except Exception:
        return None

    # The docstrings from `collections.abc` are better than those from `typing`,
    # so if the runtime object is a `typing`-module alias, return the class from
    # `collections.abc` that it's aliasing instead.
    #
    # ... with one exception to the exceptions: `typing.Callable`
    if (
        isinstance(runtime, type(typing.Mapping))
        and runtime is not typing.Callable  # type: ignore[comparison-overlap, redundant-expr]
        and runtime.__origin__.__module__ == "collections.abc"  # type: ignore[attr-defined]
    ):
        runtime = runtime.__origin__  # type: ignore[attr-defined]

    return RuntimeValue(inner=runtime)


def transform(
    source: str,
    runtime_module: types.ModuleType,
    *,
    module_name: str,
    stub_file_path: Path,
    typeshed_client_context: typeshed_client.SearchContext,
    blacklisted_objects: frozenset[str],
) -> str:
    """Add runtime docstrings to stub source and return the transformed source."""
    parsed_module = ast.parse(source)

    if runtime_module.__doc__ and ast.get_docstring(parsed_module, clean=False) is None:
        docstring = triple_quoted_docstring(runtime_module.__doc__) + "\n"
        source = docstring + source if source.strip() else docstring
        parsed_module = ast.parse(source)

    transformer = DocstringAdder(
        source=source,
        parsed_module=parsed_module,
        module_name=module_name,
        runtime_module=runtime_module,
        stub_file_path=stub_file_path,
        typeshed_client_context=typeshed_client_context,
        blacklisted_objects=blacklisted_objects,
    )
    return transformer.transform()


def add_docstrings_to_stub(
    module_name: str,
    path: Path,
    context: typeshed_client.SearchContext,
    blacklisted_objects: frozenset[str],
) -> None:
    """Add docstrings to a stub module and all functions/classes in it."""

    print(f"Processing {module_name}... ", flush=True)
    try:
        # Redirect stdout when importing modules to avoid noisy output from modules like `this`
        with contextlib.redirect_stdout(io.StringIO()):
            runtime_module = importlib.import_module(module_name)
    except KeyboardInterrupt:
        raise
    # `importlib.import_module("multiprocessing.popen_fork")` crashes with AttributeError on Windows
    # Trying to import serial.__main__ for typeshed's pyserial package will raise SystemExit
    except BaseException as e:
        log(f'Could not import {module_name}: {type(e).__name__}: "{e}"')
        return

    stub_source = path.read_text(encoding="utf-8")
    try:
        new_module = transform(
            stub_source,
            runtime_module,
            module_name=module_name,
            stub_file_path=path,
            typeshed_client_context=context,
            blacklisted_objects=blacklisted_objects,
        )
    except SyntaxError as e:
        log(f"Could not parse '{module_name}' at {path}: {e}")
        return
    path.write_text(new_module, encoding="utf-8")

    check_no_destructive_changes(
        path=path, previous_stub=stub_source, new_stub=new_module
    )


def assert_asts_match(old: ast.AST, new: ast.AST) -> None:
    """Check that two ASTs are equivalent, except for changes we choose to ignore.

    This approach is inspired by Black's AST safety check,
    found in https://github.com/psf/black/blob/f4926ace179123942d5713a11196e4a4afae1d2b/src/black/parsing.py.

    `RuntimeError` is raised if the ASTs are not equivalent.
    """
    if type(old) is not type(new):
        raise _make_safety_error(
            old, new, type(old).__name__, type(new).__name__, "AST node types differ"
        )
    # We don't use ast.iter_fields() here because it ignores fields that don't exist on the node,
    # so if we use it we could theoretically miss discrepancies where an attribute exists on new
    # but not old.
    for field_name in old._fields:
        old_field = getattr(old, field_name)
        new_field = getattr(new, field_name)

        # Allow docstrings to have been added to the body
        if field_name in {"body", "orelse"}:
            # Allow replacing a body that only has `...` in it with
            # a body that only has a docstring in it
            match (old_field, new_field):
                case (
                    [ast.Expr(value=ast.Constant(value=types.EllipsisType()))],
                    [ast.Expr(value=ast.Constant(value=str()))],
                ):
                    continue

            old_field_iter = iter(old_field)
            new_field_iter = iter(new_field)
            new_field = []
            next_old = next(old_field_iter, None)
            next_new = next(new_field_iter, None)

            while next_old is not None:
                match (next_old, next_new):
                    case (_, None):
                        raise _make_safety_error(
                            old,
                            new,
                            type(old).__name__,
                            type(new).__name__,
                            "Nodes appear to have been removed from the body of a function/class/module/if-else",
                        )
                    case (ast.Expr(value=ast.Constant(value=str())), _):
                        # if there was already a docstring, we expect it to have been preserved
                        new_field.append(next_new)
                        next_old = next(old_field_iter, None)
                        next_new = next(new_field_iter, None)
                    case (_, ast.Expr(value=ast.Constant(value=str()))):
                        # Allow new docstrings to have been added
                        # if there wasn't there one before
                        next_new = next(new_field_iter, None)
                    case _:
                        # if there wasn't a docstring previously and there isn't one being added,
                        # we expect the nodes to match
                        new_field.append(next_new)
                        next_old = next(old_field_iter, None)
                        next_new = next(new_field_iter, None)

        _assert_ast_fields_match(old, new, old_field, new_field)


def _make_safety_error(
    old: ast.AST, new: ast.AST, old_value: object, new_value: object, message: str
) -> RuntimeError:
    def explain(node: ast.AST) -> str:
        return f" (at line {node.lineno})" if hasattr(node, "lineno") else ""

    raise RuntimeError(
        f"{message}: {old_value}{explain(old)} != {new_value}{explain(new)}"
    )


_SCALAR_TYPES = (float, int, str, bytes, complex, type(None), type(...))


def _assert_ast_fields_match(
    old_container: ast.AST, new_container: ast.AST, old_value: object, new_value: object
) -> None:
    if isinstance(old_value, list) and isinstance(new_value, list):
        if len(old_value) != len(new_value):
            raise _make_safety_error(
                old_container,
                new_container,
                len(old_value),
                len(new_value),
                "AST node lists differ in length",
            )
        for old_item, new_item in zip(old_value, new_value, strict=True):
            _assert_ast_fields_match(old_container, new_container, old_item, new_item)
    elif isinstance(old_value, _SCALAR_TYPES) and isinstance(new_value, _SCALAR_TYPES):
        if type(old_value) is not type(new_value):
            raise _make_safety_error(
                old_container,
                new_container,
                type(old_value).__name__,
                type(new_value).__name__,
                "AST node types differ",
            )
        if old_value != new_value:
            raise _make_safety_error(
                old_container,
                new_container,
                old_value,
                new_value,
                "AST node values differ",
            )
    elif isinstance(old_value, ast.AST) and isinstance(new_value, ast.AST):
        assert_asts_match(old_value, new_value)
    else:
        raise _make_safety_error(
            old_container,
            new_container,
            type(old_value).__name__,
            type(new_value).__name__,
            "AST node values differ in type",
        )


def check_no_destructive_changes(path: Path, previous_stub: str, new_stub: str) -> None:
    """Check that docstring-adder has not made any destructive changes to the stub file."""
    previous_ast = ast.parse(previous_stub)

    try:
        new_ast = ast.parse(new_stub)
    except Exception:
        message = f"\nERROR: new stub file at {path} cannot be parsed/visited\n"
        print(colored(message, "red"))
        raise

    try:
        assert_asts_match(previous_ast, new_ast)
    except RuntimeError:
        message = f"\nERROR: new stub file at {path} has destructive changes\n"
        print(colored(message, "red"))
        raise


def install_typeshed_packages(typeshed_paths: Sequence[Path]) -> None:
    """Install the runtime packages corresponding to the given typeshed packages.

    The function takes a list of paths pointing to source definitions of typeshed packages.
    For each path passed, it will:
    - Install the runtime package corresponding to the typeshed package
    - Install any additional "stubtest requirements" specified in the package's
      METADATA.toml file. These are additional packages that are installed in typeshed's
      CI when running the stubtest tool for the stubs package. Often, certain submodules
      will not be importable at runtime unless these additional packages are installed.

    The packages will be directly installed into the Python environment
    that docstring-adder is being run in.
    """
    to_install: list[str] = []
    for path in typeshed_paths:
        metadata_path = path / "METADATA.toml"
        if not metadata_path.exists():
            print(f"{path} does not look like a typeshed package", file=sys.stderr)
            sys.exit(1)
        metadata_bytes = metadata_path.read_text(encoding="utf-8")
        metadata = tomli.loads(metadata_bytes)
        version = metadata["version"].lstrip("~=")
        to_install.append(f"{path.name}=={version}")
        stubtest_requirements = (
            metadata
            .get("tool", {})
            .get("stubtest", {})
            .get("stubtest_requirements", [])
        )
        to_install.extend(stubtest_requirements)
    if to_install:
        command = ["uv", "pip", "install", "--python", sys.executable, *to_install]
        print(f"Running install command: {' '.join(command)}")
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError:
            sys.exit(10)


# A hardcoded list of stdlib modules to skip
# This is separate to the --blacklists argument on the command line,
# which is for individual functions/methods/variables to skip
#
# * `_typeshed` doesn't exist at runtime; no point trying to add docstrings to it
# * `antigravity` exists at runtime but it's annoying to have the browser open up every time
# * `__main__` exists at runtime but will just reflect details of how docstring-adder itself was run.
# * `sys._monitoring` exists as `sys.monitoring` at runtime, but none of the APIs in that module
#   have docstrings, so no point trying to add them.
STDLIB_MODULE_BLACKLIST = frozenset({
    "_typeshed/*.pyi",
    "antigravity.pyi",
    "__main__.pyi",
    "sys/_monitoring.pyi",
})

STDLIB_OBJECT_BLACKLIST = frozenset({
    # On older Python versions, `enum.auto.value` is just an instance of `object`;
    # we don't want docstring-adder to add the docstring from `object` to it.
    "enum.auto.value"
})


def load_blacklist(path: Path) -> frozenset[str]:
    """Load a "blacklist" file.

    A "blacklist" file is text file containing a list of APIs
    that docstring-adder should skip when adding docstrings to stubs.
    """
    with path.open() as f:
        entries = frozenset(line.split("#")[0].strip() for line in f)
    return entries - {""}


def _main(cli_args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionRichHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--stdlib-path",
        help=(
            "Path to typeshed's stdlib directory. If given, we will add docstrings to"
            " stubs in this directory."
        ),
    )
    parser.add_argument(
        "-p",
        "--packages",
        nargs="+",
        help=(
            "List of stub package root paths to add docstrings to. We will add docstrings "
            "to these stubs. The runtime package must be installed."
        ),
        default=(),
    )
    parser.add_argument(
        "-t",
        "--typeshed-packages",
        nargs="+",
        help=(
            "List of typeshed packages to add docstrings to. WARNING: We will install the package locally. "
            "If installation of packages fails, we will exit with code 10."
        ),
        default=(),
    )
    parser.add_argument(
        "-b",
        "--blacklists",
        nargs="+",
        help=(
            "List of paths pointing to 'blacklist files', which can be used to specify functions/classes "
            "that docstring-adder should skip trying to add docstrings to."
        ),
        default=(),
    )
    args = parser.parse_args(cli_args)

    stdlib_path = Path(args.stdlib_path) if args.stdlib_path else None
    if stdlib_path is not None and not (
        stdlib_path.is_dir() and (stdlib_path / "VERSIONS").is_file()
    ):
        parser.error(f'"{stdlib_path}" does not point to a valid stdlib directory')

    typeshed_paths = [Path(p) for p in args.typeshed_packages]
    install_typeshed_packages(typeshed_paths)
    package_paths = [Path(p) for p in args.packages]
    all_package_paths = package_paths + typeshed_paths
    search_paths = [*dict.fromkeys(p.parent for p in package_paths), *typeshed_paths]

    combined_blacklist = frozenset(
        chain.from_iterable(load_blacklist(Path(path)) for path in args.blacklists)
    )
    stdlib_blacklist = combined_blacklist | STDLIB_OBJECT_BLACKLIST
    context = typeshed_client.get_search_context(
        typeshed=stdlib_path, search_path=search_paths, version=sys.version_info[:2]
    )

    codemodded_stubs = 0

    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and path.is_relative_to(stdlib_path):
            if any(
                path.relative_to(stdlib_path).match(pattern)
                for pattern in STDLIB_MODULE_BLACKLIST
            ):
                log(f"Skipping {module}: blacklisted module")
                continue
            else:
                add_docstrings_to_stub(module, path, context, stdlib_blacklist)
        elif any(path.is_relative_to(p) for p in all_package_paths):
            add_docstrings_to_stub(module, path, context, combined_blacklist)
        else:
            continue
        codemodded_stubs += 1

    if codemodded_stubs == 0:
        m = "\n--- ERROR: Didn't find any stubs to codemod for the passed packages ---"
        print(colored(m, "red"))
        sys.exit(1)
    else:
        m = "\n--- Successfully completed the codemod ---"
        print(colored(m, "green"))
        sys.exit(0)


if __name__ == "__main__":
    _main()
