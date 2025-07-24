"""Tool to add docstrings to stubs."""

from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import importlib
import inspect
import io
import subprocess
import sys
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import NamedTuple, TypeVar

import libcst
import tomli
import typeshed_client
from termcolor import colored


def log(*objects: object) -> None:
    print(colored(" ".join(map(str, objects)), "yellow"))


DocumentableT = TypeVar("DocumentableT", libcst.ClassDef, libcst.FunctionDef)
SuiteT = TypeVar("SuiteT", libcst.Module, libcst.IndentedBlock)


def triple_quoted_docstring(content: str) -> str:
    content = content.replace("\\", "\\\\")
    docstring = f'"""{content}'

    # For a single-line docstring, this can result in funny things like:
    #
    # ```py
    # def foo():
    #     """A docstring.
    #     """
    # ```
    #
    # But we don't need to worry about that here: Black sorts that out for us and turns it into:
    #
    # ```py
    # def foo():
    #     """A docstring."""
    # ```
    if not docstring.strip(" \t").endswith("\n"):
        docstring += "\n"

    return docstring + '"""'


@dataclass
class RuntimeValue:
    inner: object

    def is_not_found(self) -> bool:
        return self.inner is NOT_FOUND


class RuntimeParent(NamedTuple):
    name: str
    value: RuntimeValue


class DocstringAdder(libcst.CSTTransformer):
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

    def maybe_mangled_name(self, name: str) -> str:
        parent = self.runtime_parents[-1]
        if not isinstance(parent.value, type):
            return name

        if name.startswith("__") and not name.endswith("__"):
            return f"_{parent.name.lstrip('_')}{name}"

        return name

    def visit_ClassDef(self, node: libcst.ClassDef) -> None:
        runtime_object = get_runtime_object_for_stub(
            runtime_parent=self.runtime_parents[-1],
            name=self.maybe_mangled_name(node.name.value),
        )
        if runtime_object is None:
            self.log_runtime_object_not_found(node.name.value)
            runtime_object = RuntimeValue(NOT_FOUND)
        self.runtime_parents.append(
            RuntimeParent(name=node.name.value, value=runtime_object)
        )

    def leave_ClassDef(
        self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef
    ) -> libcst.ClassDef:
        runtime_class = self.runtime_parents.pop().value
        if runtime_class.is_not_found():
            return original_node
        else:
            return self.document_class_or_function(
                updated_node=updated_node, runtime_object=runtime_class
            )

    def object_fullname(self, final_part: str) -> str:
        parent_fullname = ".".join(parent.name for parent in self.runtime_parents)
        mangled_name = self.maybe_mangled_name(final_part)
        return f"{parent_fullname}.{mangled_name}"

    def log_runtime_object_not_found(self, name: str) -> None:
        log(f"Could not find {self.object_fullname(name)} at runtime")

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        runtime_parent = self.runtime_parents[-1]
        if runtime_parent.value.is_not_found():
            return original_node
        runtime_object = get_runtime_object_for_stub(
            runtime_parent=runtime_parent,
            name=self.maybe_mangled_name(updated_node.name.value),
        )
        if runtime_object is None:
            self.log_runtime_object_not_found(updated_node.name.value)
            return original_node
        return self.document_class_or_function(
            updated_node=updated_node, runtime_object=runtime_object
        )

    def _leave_Suite(self, original_node: SuiteT, updated_node: SuiteT) -> SuiteT:
        assert len(original_node.body) == len(updated_node.body)

        body: list[libcst.BaseStatement] = []
        seen: set[str] = set()

        for original, updated in zip(original_node.body, updated_node.body):
            if not isinstance(updated, libcst.FunctionDef):
                body.append(updated)
                continue
            # If there are multiple functions with the same name in an indented block,
            # it's probably an overloaded function or the `@setter` for a property.
            # Only add a docstring for the first definition.
            if updated.name.value in seen:
                body.append(original)
                continue
            seen.add(updated.name.value)
            body.append(updated)

        return updated_node.with_changes(body=body)

    def leave_IndentedBlock(
        self, original_node: libcst.IndentedBlock, updated_node: libcst.IndentedBlock
    ) -> libcst.IndentedBlock:
        return self._leave_Suite(original_node, updated_node)

    def leave_Module(
        self, original_node: libcst.Module, updated_node: libcst.Module
    ) -> libcst.Module:
        return self._leave_Suite(original_node, updated_node)

    def leave_If(self, original_node: libcst.If, updated_node: libcst.If) -> libcst.If:
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
        object_fullname = self.object_fullname(updated_node.name.value)
        if object_fullname in self.blacklisted_objects:
            return updated_node

        if updated_node.get_docstring() is not None:
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
            and updated_node.name.value in object.__dict__
        ):
            method_docstring_on_object = get_runtime_docstring(
                runtime=RuntimeValue(inner=object.__dict__[updated_node.name.value])
            )
            if docstring == method_docstring_on_object:
                return updated_node

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
                # but preserve `# type: ignore` comments
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
    runtime_object = runtime.inner

    try:
        # Don't use `inspect.getdoc()` here: it returns the docstring from superclasses
        # if the docstring has not been overridden on a subclass, and that's not what we want.
        runtime_docstring = runtime_object.__doc__
    except Exception:
        return None

    if not isinstance(runtime_docstring, str):
        return None

    if runtime_object is not staticmethod:
        assert runtime_docstring != staticmethod.__doc__, runtime

    if runtime_object is not classmethod:
        assert runtime_docstring != classmethod.__doc__, runtime

    if runtime_object is not property:
        assert runtime_docstring != property.__doc__, runtime

    return inspect.cleandoc(runtime_docstring)


class NOT_FOUND: ...


def get_runtime_object_for_stub(
    runtime_parent: RuntimeParent, name: str
) -> RuntimeValue | None:
    # Some `sys`-module APIs are weird.
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
        if (
            sys.version_info < (3, 10)
            and not isinstance(runtime, type)
            and hasattr(runtime, "__func__")
        ):
            runtime = runtime.__func__
    # Some getattr() calls raise TypeError, or something even more exotic
    except Exception:
        return None

    # The docstrings from `collections.abc` are better than those from `typing`.
    if isinstance(runtime, type(typing.Mapping)):
        runtime = runtime.__origin__  # type: ignore[attr-defined]

    return RuntimeValue(inner=runtime)


def add_docstrings_to_stub(
    module_name: str,
    context: typeshed_client.SearchContext,
    blacklisted_objects: frozenset[str],
) -> None:
    """Add docstrings a stub module and all functions/classes in it."""

    print(f"Processing {module_name}... ", flush=True)
    path = typeshed_client.get_stub_file(module_name, search_context=context)
    if path is None:
        raise ValueError(f"Could not find stub for {module_name}")
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
    new_module = parsed_module.visit(transformer).code
    path.write_text(new_module, encoding="utf-8")

    check_no_destructive_changes(
        path=path, previous_stub=stub_source, new_stub=new_module
    )


class SanityChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: collections.Counter[str] = collections.Counter()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names[node.name] += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names[node.name] += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names[node.name] += 1
        self.generic_visit(node)


def check_no_destructive_changes(path: Path, previous_stub: str, new_stub: str) -> None:
    """Check that the new stub does not contain any destructive changes."""
    previous_ast = ast.parse(previous_stub)
    previous_checker = SanityChecker()
    previous_checker.visit(previous_ast)

    try:
        new_ast = ast.parse(new_stub)
        new_checker = SanityChecker()
        new_checker.visit(new_ast)
    except Exception:
        message = f"\nERROR: new stub file at {path} cannot be parsed/visited\n"
        print(colored(message, "red"))
        raise

    assert previous_checker.names == new_checker.names, (
        f"Destructive changes appear to have been made to {path}"
    )


def install_typeshed_packages(typeshed_paths: Sequence[Path]) -> None:
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
    if to_install:
        command = ["uv", "pip", "install", "--python", sys.executable, *to_install]
        print(f"Running install command: {' '.join(command)}")
        subprocess.check_call(command)


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
    with path.open() as f:
        entries = frozenset(line.split("#")[0].strip() for line in f)
    return entries - {""}


def main() -> None:
    parser = argparse.ArgumentParser()
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
            "List of typeshed packages to add docstrings to. WARNING: We will install the package locally."
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
    parser.add_argument(
        "-z",
        "--exit-zero",
        action="store_true",
        help="Exit with code 0 even if there were errors.",
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
    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and path.is_relative_to(stdlib_path):
            if any(
                path.relative_to(stdlib_path).match(pattern)
                for pattern in STDLIB_MODULE_BLACKLIST
            ):
                log(f"Skipping {module}: blacklisted module")
                continue
            else:
                add_docstrings_to_stub(module, context, stdlib_blacklist)
        elif any(path.is_relative_to(p) for p in package_paths):
            add_docstrings_to_stub(module, context, combined_blacklist)
    m = "\n--- Successfully codemodded typeshed ---"
    print(colored(m, "green"))
    sys.exit(0)


if __name__ == "__main__":
    main()
