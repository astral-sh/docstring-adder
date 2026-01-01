"""Tool to add docstrings to stubs.

The tool is a libcst-based codemod. Given a path to a typeshed `stdlib` directory or a
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
   e. The modified stub file is written back to disk.
   f. An AST safety check is performed to ensure that the modified stub file is still
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
- Because it is a libcst-based codemod, it should not make spurious changes to formatting
  or comments in the stub file. `type: ignore` comments should be preserved; mypy and
  other type checkers should still be able to type-check the stub file after
  docstring-adder has run.
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
  similarly, if it is being run on Python 3.9, it should not add any docstrings to
  definitions inside `if sys.version_info >= (3, 10)` branches. Fundamentally, the tool
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
import contextlib
import importlib
import inspect
import io
import itertools
import subprocess
import sys
import textwrap
import types
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TypeVar

import libcst
import tomli
import typeshed_client
from rich_argparse import RawDescriptionRichHelpFormatter
from termcolor import colored
from typing_extensions import override


def log(*objects: object) -> None:
    """Log a warning to the terminal."""
    print(colored(" ".join(map(str, objects)), "yellow"))


# Type variable representing a node that could have a docstring added to it.
DocumentableT = TypeVar("DocumentableT", libcst.ClassDef, libcst.FunctionDef)


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
    if quote == escaped_string[-1]:
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


class DocstringAdder(libcst.CSTTransformer):
    """Visitor to add docstrings to a stub file.

    The visitor is a `libcst.CSTTransformer` that recursively attempts to adds
    docstrings to all reachable class/function definitions inside a given stub file.
    """

    def __init__(
        self,
        *,
        module_name: str,
        runtime_module: types.ModuleType,
        stub_file_path: Path,
        typeshed_client_context: typeshed_client.SearchContext,
        blacklisted_objects: frozenset[str],
    ) -> None:
        self.runtime_parents: list[RuntimeParent] = [
            RuntimeParent(name=module_name, value=RuntimeValue(inner=runtime_module))
        ]
        self.stub_file_path: Path = stub_file_path
        self.typeshed_client_context: typeshed_client.SearchContext = (
            typeshed_client_context
        )
        self.blacklisted_objects: frozenset[str] = blacklisted_objects

        # A stack of sets. Each set corresponds to an `IndentedBlock` libcst node.
        # For each nested `IndentedBlock`, we keep track of the names of functions
        # that we've already visited (and, possibly, added docstrings to).
        #
        # We start off with a single empty set, which corresponds to the module-level
        # namespace. The module-level namespace is not represented by an `IndentedBlock`
        # in libcst's CST, but for our purposes we treat the module-level namespace
        # just like any indented block.
        self.suite_visitation_stack: list[set[str]] = [set()]

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
        if not isinstance(parent.value, type):
            return name

        if name.startswith("__") and not name.endswith("__"):
            return f"_{parent.name.lstrip('_')}{name}"

        return name

    @override
    def visit_ClassDef(self, node: libcst.ClassDef) -> None:
        """Visit a class definition node in the stub file.

        This hook is called before any sub-statements and sub-expressions inside the
        class definition are visited or transformed. This allows us to retrieve the
        runtime object representing the class, and append it to the stack of runtime
        parents.
        """
        runtime_object = get_runtime_object_for_stub(
            name=self.maybe_mangled_name(node.name.value),
            runtime_parent=self.runtime_parents[-1],
        )
        if runtime_object is None:
            self._log_runtime_object_not_found(node.name.value)
            runtime_object = RuntimeValue(NOT_FOUND)
        self.runtime_parents.append(
            RuntimeParent(name=node.name.value, value=runtime_object)
        )

    @override
    def leave_ClassDef(
        self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef
    ) -> libcst.ClassDef:
        """Attempt to add a docstring to the class definition.

        This hook is called after all sub-statements and sub-expressions inside the
        class definition have been visited and transformed.
        """

        runtime_class = self.runtime_parents.pop().value
        if runtime_class.is_not_found():
            return original_node
        else:
            return self.document_class_or_function(
                updated_node=updated_node, runtime_object=runtime_class
            )

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

    def _log_runtime_object_not_found(self, name: str) -> None:
        log(f"Could not find {self.object_fullname(name)} at runtime")

    @override
    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        """Attempt to add a docstring to the function definition."""

        # If there are multiple functions with the same name in an indented block,
        # it's probably an overloaded function or the `@setter` for a property.
        # Only add a docstring for the first definition.
        if original_node.name.value in self.suite_visitation_stack[-1]:
            return original_node
        self.suite_visitation_stack[-1].add(original_node.name.value)

        runtime_parent = self.runtime_parents[-1]

        if runtime_parent.value.is_not_found():
            return original_node

        runtime_object = get_runtime_object_for_stub(
            name=self.maybe_mangled_name(updated_node.name.value),
            runtime_parent=runtime_parent,
        )

        if runtime_object is None:
            self._log_runtime_object_not_found(updated_node.name.value)
            return original_node

        return self.document_class_or_function(
            updated_node=updated_node, runtime_object=runtime_object
        )

    @override
    def visit_IndentedBlock(self, node: libcst.IndentedBlock) -> None:
        """Hook called before visiting an `IndentedBlock` node."""
        self.suite_visitation_stack.append(set())

    @override
    def leave_IndentedBlock(
        self, original_node: libcst.IndentedBlock, updated_node: libcst.IndentedBlock
    ) -> libcst.IndentedBlock:
        """Hook called after an `IndentedBlock` node has been visited and, possibly, modified."""
        self.suite_visitation_stack.pop()

        # Don't add attribute docstrings to fields in `NamedTuple` classes.
        # The generated docstrings for the properties in these classes
        # are always things like "Alias for field number 0", which clutter
        # the stubs and aren't useful.
        runtime_parent = self.runtime_parents[-1].value.inner
        if (
            isinstance(runtime_parent, type)
            and issubclass(runtime_parent, tuple)
            and hasattr(runtime_parent, "_fields")
        ):
            return updated_node

        return updated_node.with_changes(
            body=add_attribute_docstrings(
                updated_node.body,
                runtime_parent=self.runtime_parents[-1],
                blacklisted_objects=self.blacklisted_objects,
                indentation=len(self.suite_visitation_stack),
            )
        )

    @override
    def leave_If(self, original_node: libcst.If, updated_node: libcst.If) -> libcst.If:
        """Hook called when the transformer leaves an `if` statement.

        All sub-statements and sub-expressions inside the `if` statement
        will already have been visited (and, possibly, modified) by the
        transformer. This method discards any changes that may have been
        made to branches of code that are unreachable on the platform
        and Python version docstring-adder is being run on.

        See the module-level docstring for why we do this, and caveats
        regarding how we evaluate the truthiness of the `if` test.

        It might theoretically be possible to add more state to the
        transformer so that we avoid making the undesirable changes in
        the first place, rather than applying the undesirable changes and
        then discarding them later on. This approach seems to work well
        enough for now, however, and is simpler.
        """
        condition_as_libcst_module = libcst.Module(
            body=[
                libcst.SimpleStatementLine(body=[libcst.Expr(value=original_node.test)])
            ]
        )
        condition_as_ast_expr = ast.parse(condition_as_libcst_module.code).body[0]
        assert isinstance(condition_as_ast_expr, ast.Expr)
        parsed_condition = typeshed_client.evaluate_expression_truthiness(
            condition_as_ast_expr.value,
            ctx=self.typeshed_client_context,
            file_path=self.stub_file_path,
        )
        assert isinstance(parsed_condition, bool)
        if parsed_condition:
            return updated_node.with_changes(orelse=original_node.orelse)
        else:
            return updated_node.with_changes(body=original_node.body)

    def document_class_or_function(
        self, updated_node: DocumentableT, runtime_object: RuntimeValue
    ) -> DocumentableT:
        """Attempt to add a docstring to a class or function definition."""

        object_fullname = self.object_fullname(updated_node.name.value)
        if object_fullname in self.blacklisted_objects:
            return updated_node

        if updated_node.get_docstring() is not None:  # ty: ignore[invalid-argument-type]
            return updated_node

        docstring = get_runtime_docstring(runtime=runtime_object)
        if docstring is None:
            return updated_node

        # E.g. we want to avoid a bajillion `__init__` docstrings that are just
        #
        # > Initialize self.  See help(type(self)) for accurate signature.
        #
        # Which is exactly what we get for `help(object.__init__)`
        if (
            # not sure why mypy thinks this is redundant...!
            isinstance(updated_node, libcst.FunctionDef)  # type: ignore[redundant-expr]
            and not isinstance(runtime_object.inner, types.ModuleType)
            and updated_node.name.value in object.__dict__
        ):
            method_docstring_on_object = get_runtime_docstring(
                runtime=RuntimeValue(inner=object.__dict__[updated_node.name.value])
            )
            if docstring == method_docstring_on_object:
                return updated_node  # ty: ignore[invalid-return-type]

        docstring_node = libcst.Expr(
            libcst.SimpleString(triple_quoted_docstring(docstring))
        )

        # If the body is just a `...`, replace it with just the docstring.
        if (
            isinstance(updated_node.body, libcst.SimpleStatementSuite)
            and len(updated_node.body.body) == 1
            and isinstance(updated_node.body.body[0], libcst.Expr)
            and isinstance(updated_node.body.body[0].value, libcst.Ellipsis)
        ):
            new_body = libcst.IndentedBlock(
                body=[libcst.SimpleStatementLine(body=[docstring_node])],
                # but preserve `type: ignore` comments
                header=updated_node.body.trailing_whitespace,
            )
        elif isinstance(updated_node.body, libcst.IndentedBlock):
            if (
                len(updated_node.body.body) == 1
                and isinstance(updated_node.body.body[0], libcst.Expr)
                and isinstance(updated_node.body.body[0].value, libcst.Ellipsis)
            ):
                new_body = updated_node.body.with_changes(
                    body=[libcst.SimpleStatementLine(body=[docstring_node])]
                )
            else:
                new_body = updated_node.body.with_changes(
                    body=list(
                        chain(
                            [libcst.SimpleStatementLine(body=[docstring_node])],
                            updated_node.body.body,
                        )
                    )
                )
        else:
            return updated_node

        return updated_node.with_changes(body=new_body)


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


SuiteItemT = TypeVar(
    "SuiteItemT",
    "libcst.SimpleStatementLine | libcst.BaseCompoundStatement",
    "libcst.BaseSmallStatement | libcst.BaseStatement",
)


def final_statement_of_if(
    node: libcst.If,
) -> libcst.BaseStatement | libcst.BaseSmallStatement:
    """Retrieve the final statement of an `if`/`elif`/`else` chain."""
    if node.orelse is None:
        return node.body.body[-1]
    if isinstance(node.orelse, libcst.Else):
        return node.orelse.body.body[-1]
    return final_statement_of_if(node.orelse)


def add_attribute_docstrings(
    body: Sequence[SuiteItemT],
    *,
    runtime_parent: RuntimeParent,
    blacklisted_objects: frozenset[str],
    indentation: int,
) -> list[SuiteItemT]:
    """Add Sphinx-style 'attribute docstrings' to assignments in a suite.

    The suite could be the body of a module, class, function,
    `if` block, `elif` block, or `else` block.
    """
    new_body: list[SuiteItemT] = []
    added_docstring_to_previous = False
    for statement, next_statement in itertools.zip_longest(body, body[1:]):
        # If we just added a docstring to the previous statement,
        # add a blank line before this statement.
        # Black will not do this for us.
        if (
            added_docstring_to_previous
            and hasattr(statement, "leading_lines")
            and isinstance(statement.leading_lines, Iterable)
            and all(
                isinstance(line, libcst.EmptyLine) and line.comment is not None
                for line in statement.leading_lines
            )
        ):
            new_body.append(
                statement.with_changes(
                    leading_lines=[libcst.EmptyLine(), *statement.leading_lines]  # type: ignore[has-type]
                )
            )
        else:
            new_body.append(statement)

        added_docstring_to_previous = False

        if isinstance(statement, libcst.If):
            final_line = final_statement_of_if(statement)
        elif isinstance(statement, (libcst.FunctionDef, libcst.ClassDef)):
            final_line = (
                final_statement_of_if(statement.body.body[-1])
                if isinstance(statement.body.body[-1], libcst.If)
                else statement.body.body[-1]
            )
        else:
            final_line = None

        if (
            final_line is not None
            and isinstance(final_line, libcst.SimpleStatementLine)
            and len(final_line.body) == 1
            and isinstance(final_line.body[0], libcst.Expr)
            and isinstance(final_line.body[0].value, libcst.SimpleString)
        ):
            added_docstring_to_previous = True
            continue

        # If it's an annotated assignment that we could potentially add a docstring to...
        if (
            isinstance(statement, libcst.SimpleStatementLine)
            and len(statement.body) == 1
            # ... and there is no docstring already present...
            and not (
                isinstance(next_statement, libcst.SimpleStatementLine)
                and len(next_statement.body) == 1
                and isinstance(next_statement.body[0], libcst.Expr)
                and isinstance(next_statement.body[0].value, libcst.SimpleString)
            )
        ):
            assignment = statement.body[0]
            if isinstance(assignment, libcst.AnnAssign):
                target = assignment.target
            elif isinstance(assignment, libcst.Assign) and len(assignment.targets) == 1:
                target = assignment.targets[0].target
            else:
                continue

            if not isinstance(target, libcst.Name):
                continue

            # ... then try to add a docstring to it.
            runtime_name = target.value
            if f"{runtime_parent.name}.{runtime_name}" in blacklisted_objects:
                continue

            runtime_value = get_runtime_object_for_stub(runtime_name, runtime_parent)

            if runtime_value is None:
                continue

            # Heuristics to avoid adding undesirable attribute docstrings.
            #
            # For example, if it's an unannotated assignment to a
            # class/function/module, or it's likely to be a type alias to a
            # class, there's no need to add an attribute docstring to the
            # variable: type checkers should pick up the docstring anyway.
            #
            # We also avoid adding docstrings to `Assign`/`AnnAssign` nodes
            # where the runtime value is a class/function that comes from
            # a different module. It's usually undesirable to add docstrings
            # for these attributes.
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
                if isinstance(assignment, libcst.Assign):
                    continue
                if assignment.value is not None:
                    continue
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
                indentation_string = " " * indentation * 4
                docstring = (
                    textwrap.indent(docstring, prefix=indentation_string)
                    + indentation_string
                ).lstrip(" \t")
            else:
                indentation_string = None

            new_body.append(
                libcst.SimpleStatementLine(
                    body=[
                        libcst.Expr(
                            libcst.SimpleString(
                                triple_quoted_docstring(
                                    docstring, indentation=indentation_string
                                )
                            )
                        )
                    ]
                )
            )
            added_docstring_to_previous = True

    return new_body


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
        # `inspect.unwrap()` doesn't do a great job for staticmethod/classmethod on Python 3.9,
        # because the `__wrapped__` attribute was added for these objects in Python 3.10.
        # Instead, retrieve the underlying (hopefully documented) function by accessing
        # the `__func__` attribute.
        if (
            sys.version_info < (3, 10)
            and not isinstance(runtime, type)
            and hasattr(runtime, "__func__")
        ):
            runtime = runtime.__func__
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
        and runtime is not typing.Callable  # type: ignore[comparison-overlap]
        and runtime.__origin__.__module__ == "collections.abc"  # type: ignore[attr-defined]
    ):
        runtime = runtime.__origin__  # type: ignore[attr-defined]

    return RuntimeValue(inner=runtime)


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
    parsed_module = libcst.parse_module(stub_source)

    if runtime_module.__doc__ and parsed_module.get_docstring() is None:
        docstring = triple_quoted_docstring(runtime_module.__doc__) + "\n"
        stub_source = docstring + stub_source if stub_source.strip() else docstring
        parsed_module = libcst.parse_module(stub_source)

    transformer = DocstringAdder(
        module_name=module_name,
        runtime_module=runtime_module,
        stub_file_path=path,
        typeshed_client_context=context,
        blacklisted_objects=blacklisted_objects,
    )
    parsed_module = parsed_module.visit(transformer)
    parsed_module = parsed_module.with_changes(
        body=add_attribute_docstrings(
            parsed_module.body,
            runtime_parent=RuntimeParent(module_name, RuntimeValue(runtime_module)),
            blacklisted_objects=blacklisted_objects,
            indentation=0,
        )
    )
    new_module = parsed_module.code
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
            if (
                len(old_field) == 1
                and isinstance(old_field[0], ast.Expr)
                and isinstance(old_field[0].value, ast.Constant)
                and old_field[0].value.value is ...
                and len(new_field) == 1
                and isinstance(new_field[0], ast.Expr)
                and isinstance(new_field[0].value, ast.Constant)
                and isinstance(new_field[0].value.value, str)
            ):
                continue

            old_field_iter = iter(old_field)
            new_field_iter = iter(new_field)
            new_field = []
            next_old = next(old_field_iter, None)
            next_new = next(new_field_iter, None)

            while next_old is not None:
                if next_new is None:
                    raise _make_safety_error(
                        old,
                        new,
                        type(old).__name__,
                        type(new).__name__,
                        "Nodes appear to have been removed from the body of a function/class/module/if-else",
                    )

                # if there was already a docstring, we expect it to have been preserved
                if (
                    isinstance(next_old, ast.Expr)
                    and isinstance(next_old.value, ast.Constant)
                    and isinstance(next_old.value.value, str)
                ):
                    new_field.append(next_new)
                    next_old = next(old_field_iter, None)
                    next_new = next(new_field_iter, None)

                # Allow new docstrings to have been added
                # if there wasn't there one before
                elif (
                    isinstance(next_new, ast.Expr)
                    and isinstance(next_new.value, ast.Constant)
                    and isinstance(next_new.value.value, str)
                ):
                    next_new = next(new_field_iter, None)

                # if there wasn't a docstring previously and there isn't one being added,
                # we expect the nodes to match
                else:
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
        for old_item, new_item in zip(old_value, new_value):
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
            metadata.get("tool", {})
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


def _main() -> None:
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
            "List of packages to add docstrings to. We will add docstrings to all stubs in"
            " these directories. The runtime package must be installed."
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
    args = parser.parse_args()

    stdlib_path = Path(args.stdlib_path) if args.stdlib_path else None
    if stdlib_path is not None and not (
        stdlib_path.is_dir() and (stdlib_path / "VERSIONS").is_file()
    ):
        parser.error(f'"{stdlib_path}" does not point to a valid stdlib directory')

    typeshed_paths = [Path(p) for p in args.typeshed_packages]
    install_typeshed_packages(typeshed_paths)
    package_paths = [Path(p) for p in args.packages] + typeshed_paths

    combined_blacklist = frozenset(
        chain.from_iterable(load_blacklist(Path(path)) for path in args.blacklists)
    )
    stdlib_blacklist = combined_blacklist | STDLIB_OBJECT_BLACKLIST
    context = typeshed_client.get_search_context(
        typeshed=stdlib_path, search_path=package_paths, version=sys.version_info[:2]
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
        elif any(path.is_relative_to(p) for p in package_paths):
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
