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
import textwrap
import types
import typing
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple

import libcst
import tomli
import typeshed_client
from termcolor import colored


def log(*objects: object) -> None:
    print(colored(" ".join(map(str, objects)), "yellow"))


def get_end_lineno(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    assert node.end_lineno is not None
    return node.end_lineno


class DocumentableObject(NamedTuple):
    """An object in a stub that could have a docstring added to it."""

    fullname: str
    stub_ast: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    runtime_object: Any


def add_docstring_to_libcst_node[T: (libcst.ClassDef, libcst.FunctionDef)](
    original_node: T,
    updated_node: T,
    object_to_codemod: libcst.ClassDef | libcst.FunctionDef,
    runtime_object: Any,
    *,
    level: int,
) -> tuple[T, bool]:
    if original_node is not object_to_codemod:
        return updated_node, False

    if updated_node.get_docstring() is not None:
        return updated_node, False

    docstring = get_runtime_docstring(runtime=runtime_object)
    if docstring is None:
        return updated_node, False

    # E.g. we want to avoid a bajillion `__init__` docstrings that are just
    #
    # > Initialize self.  See help(type(self)) for accurate signature.
    #
    # Which is exactly what we get for `help(object.__init__)`
    if (
        level > 1
        # not sure why mypy thinks this is redundant...!
        and isinstance(updated_node, libcst.FunctionDef)  # type: ignore[redundant-expr]
        and updated_node.name.value in object.__dict__
    ):
        method_docstring_on_object = get_runtime_docstring(
            runtime=object.__dict__[updated_node.name.value]
        )
        if docstring == method_docstring_on_object:
            return updated_node, False

    indentation = " " * 4 * level
    indented_docstring = f'"""\n{textwrap.indent(docstring.strip().replace("\\", "\\\\"), indentation)}\n{indentation}"""'
    docstring_node = libcst.Expr(libcst.SimpleString(indented_docstring))

    match updated_node.body:
        # If the body is just a `...`, replace it with just the docstring.
        case libcst.SimpleStatementSuite(body=[libcst.Expr(value=libcst.Ellipsis())]):
            new_body = libcst.IndentedBlock(
                body=[libcst.SimpleStatementLine(body=[docstring_node])],
                # but preserve `# type: ignore` comments
                header=updated_node.body.trailing_whitespace,
            )

        case libcst.IndentedBlock(
            body=[
                libcst.SimpleStatementLine(body=[libcst.Expr(value=libcst.Ellipsis())])
            ]
        ):
            new_body = updated_node.body.with_changes(
                body=[libcst.SimpleStatementLine(body=[docstring_node])]
            )

        # Otherwise, add the docstring to the top of the suite
        case libcst.IndentedBlock():
            new_body = updated_node.body.with_changes(
                body=list(
                    chain(
                        [libcst.SimpleStatementLine(body=[docstring_node])],
                        updated_node.body.body,
                    )
                )
            )

        case _:
            return updated_node, False

    return updated_node.with_changes(body=new_body), True


@dataclass(kw_only=True)
class DocstringAdder(libcst.CSTTransformer):
    fullname: str
    object_to_codemod: libcst.ClassDef | libcst.FunctionDef
    runtime_object: Any
    level: int
    added_docstrings: int = 0

    def leave_ClassDef(
        self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef
    ) -> libcst.ClassDef:
        """Add a docstring to a class definition."""

        transformed_node, added = add_docstring_to_libcst_node(
            original_node=original_node,
            updated_node=updated_node,
            object_to_codemod=self.object_to_codemod,
            runtime_object=self.runtime_object,
            level=self.level,
        )
        self.added_docstrings += added

        body: list[libcst.BaseStatement | libcst.BaseSmallStatement] = []
        seen_children: set[str] = set()

        for item in transformed_node.body.body:
            if not isinstance(item, (libcst.ClassDef, libcst.FunctionDef)):
                body.append(item)
                continue

            # For overloaded methods, only add the docstring to the first overload
            if (
                isinstance(item, libcst.FunctionDef)
                and item.name.value in seen_children
            ):
                body.append(item)
                continue

            seen_children.add(item.name.value)
            maybe_mangled_child_name = maybe_mangle_name(item.name.value, self.fullname)
            runtime = get_runtime_object_for_stub(
                runtime_parent=self.runtime_object, name=maybe_mangled_child_name
            )
            if runtime is NOT_FOUND:
                log(
                    f"Could not find {self.fullname}.{maybe_mangled_child_name} at runtime"
                )
                body.append(item)
                continue
            visitor = DocstringAdder(
                fullname=f"{self.fullname}.{maybe_mangled_child_name}",
                object_to_codemod=item,
                runtime_object=runtime,
                level=self.level + 1,
            )
            transformed_item = item.visit(visitor)
            self.added_docstrings += visitor.added_docstrings
            assert isinstance(transformed_item, type(item))
            assert isinstance(transformed_item, (libcst.ClassDef, libcst.FunctionDef))
            body.append(transformed_item)

        transformed_node = transformed_node.with_changes(
            body=transformed_node.body.with_changes(body=body)
        )

        return transformed_node

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        """Add a docstring to a function definition."""

        transformed_node, added = add_docstring_to_libcst_node(
            original_node=original_node,
            updated_node=updated_node,
            object_to_codemod=self.object_to_codemod,
            runtime_object=self.runtime_object,
            level=self.level,
        )
        self.added_docstrings += added
        return transformed_node


@dataclass(kw_only=True)
class FunctionDocstringAdder(libcst.CSTTransformer):
    function_to_codemod: libcst.Name
    runtime_docstring: str
    added_docstring: bool = False


def get_runtime_docstring(runtime: Any) -> str | None:
    try:
        # Don't use `inspect.getdoc()` here: it returns the docstring from superclasses
        # if the docstring has not been overridden on a subclass, and that's not what we want.
        runtime_docstring = runtime.__doc__
    except Exception:
        return None

    if not isinstance(runtime_docstring, str):
        return None

    return inspect.cleandoc(runtime_docstring)


def add_docstring_to_stub_object(
    stub_lines: list[str], documentable_object: DocumentableObject
) -> dict[int, list[str]]:
    start_lineno = documentable_object.stub_ast.lineno - 1
    end_lineno = get_end_lineno(documentable_object.stub_ast)
    lines = stub_lines[start_lineno:end_lineno]
    cst = libcst.parse_statement(textwrap.dedent("\n".join(lines)))
    assert isinstance(cst, (libcst.FunctionDef, libcst.ClassDef))
    if cst.get_docstring() is not None:
        return {}

    visitor = DocstringAdder(
        object_to_codemod=cst,
        fullname=documentable_object.fullname,
        runtime_object=documentable_object.runtime_object,
        level=1,
    )
    modified = cst.visit(visitor)
    assert isinstance(modified, type(cst))
    assert isinstance(modified, (libcst.FunctionDef, libcst.ClassDef))
    indentation = len(lines[0]) - len(lines[0].lstrip())
    new_code = textwrap.indent(libcst.Module(body=[modified]).code, " " * indentation)
    output_dict = {start_lineno: new_code.splitlines()}
    for i in range(start_lineno + 1, end_lineno):
        output_dict[i] = []
    return output_dict


class NOT_FOUND: ...


def get_runtime_object_for_stub(
    runtime_parent: type | types.ModuleType, name: str
) -> Any:
    try:
        runtime = inspect.getattr_static(runtime_parent, name)
    # Some getattr() calls raise TypeError, or something even more exotic
    except Exception:
        return NOT_FOUND

    # The docstrings from `collections.abc` are better than those from `typing`.
    if isinstance(runtime, type(typing.Mapping)):
        runtime = runtime.__origin__  # type: ignore[attr-defined]

    return runtime


def maybe_mangle_name(name: str, parent_fullname: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        unmangled_parent_name = parent_fullname.split(".")[-1]
        return f"_{unmangled_parent_name.lstrip('_')}{name}"

    return name


def gather_documentable_objects(
    node: typeshed_client.NameInfo,
    name: str,
    fullname: str,
    runtime_parent: type | types.ModuleType,
    blacklisted_objects: frozenset[str],
) -> Iterator[DocumentableObject]:
    """Return an iterator of all names in a stub that could potentially have docstrings added to them."""

    interesting_classes = (
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        typeshed_client.OverloadedName,
    )

    if not isinstance(node.ast, interesting_classes):
        return

    runtime = get_runtime_object_for_stub(runtime_parent, name)
    if runtime is NOT_FOUND:
        log("Could not find", fullname, "at runtime")
        return

    if isinstance(node.ast, ast.ClassDef):
        if fullname in blacklisted_objects:
            log(f"Skipping {fullname}: blacklisted object")
        else:
            yield DocumentableObject(fullname, node.ast, runtime)
            return

        # Only recurse into the class if the class itself is blacklisted;
        # otherwise, subnodes are handled by the libcst transformer
        child_nodes = node.child_nodes or {}

        for child_name, child_node in child_nodes.items():
            maybe_mangled_child_name = maybe_mangle_name(child_name, fullname)

            yield from gather_documentable_objects(
                node=child_node,
                name=maybe_mangled_child_name,
                fullname=f"{fullname}.{child_name}",
                runtime_parent=runtime,
                blacklisted_objects=blacklisted_objects,
            )

        return

    if fullname in blacklisted_objects:
        log(f"Skipping {fullname}: blacklisted object")
        return

    if isinstance(node.ast, typeshed_client.OverloadedName):
        if isinstance(node.ast.definitions[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield DocumentableObject(fullname, node.ast.definitions[0], runtime)
    else:
        yield DocumentableObject(fullname, node.ast, runtime)


def add_docstrings_to_stub(
    module_name: str,
    context: typeshed_client.finder.SearchContext,
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

    stub_source = path.read_text()
    parsed_module = libcst.parse_module(stub_source)
    stub_lines = stub_source.splitlines()
    replacement_lines: dict[int, list[str]] = {}

    if runtime_module.__doc__ and parsed_module.get_docstring() is None:
        replacement_lines[0] = list(
            chain(
                ['"""'],
                runtime_module.__doc__.strip().replace("\\", "\\\\").splitlines(),
                ['"""'],
            )
        )
        if stub_lines:
            replacement_lines[0] += ["", stub_lines[0]]

    stub_names = typeshed_client.get_stub_names(module_name, search_context=context)
    if stub_names is None:
        raise ValueError(f"Could not find stub for {module_name}")

    original_objects: list[str | None] = []

    for name, info in stub_names.items():
        objects = gather_documentable_objects(
            node=info,
            name=name,
            fullname=f"{module_name}.{name}",
            runtime_parent=runtime_module,
            blacklisted_objects=blacklisted_objects,
        )

        for documentable_object in objects:
            original_objects.append(
                getattr(documentable_object.runtime_object, "__name__", None)
            )
            new_lines = add_docstring_to_stub_object(stub_lines, documentable_object)
            replacement_lines.update(new_lines)

    new_module = ""
    for i, line in enumerate(stub_lines):
        if i in replacement_lines:
            for new_line in replacement_lines[i]:
                new_module += f"{new_line}\n"
        else:
            new_module += f"{line}\n"
    path.write_text(new_module)

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
    new_ast = ast.parse(new_stub)

    previous_checker = SanityChecker()
    previous_checker.visit(previous_ast)

    new_checker = SanityChecker()
    new_checker.visit(new_ast)

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
        metadata_bytes = metadata_path.read_text()
        metadata = tomli.loads(metadata_bytes)
        version = metadata["version"]
        to_install.append(f"{path.name}=={version}")
    if to_install:
        command = [sys.executable, "-m", "pip", "install", *to_install]
        print(f"Running install command: {' '.join(command)}")
        subprocess.check_call(command)


# A hardcoded list of stdlib modules to skip
# This is separate to the --blacklists argument on the command line,
# which is for individual functions/methods/variables to skip
#
# * `_typeshed` doesn't exist at runtime; no point trying to add docstrings to it
# * `antigravity` exists at runtime but it's annoying to have the browser open up every time
# * `__main__` exists at runtime but will just reflect details of how docstring-adder itself was run.
STDLIB_MODULE_BLACKLIST = ("_typeshed/*.pyi", "antigravity.pyi", "__main__.pyi")


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
            "List of paths pointing to 'blacklist files',"
            " which can be used to specify functions/classes that docstring-adder should skip"
            " trying to add docstrings to. Note: if the name of a class is included"
            " in a blacklist, all methods within the class will be skipped as well as the class."
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
    context = typeshed_client.finder.get_search_context(
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
                add_docstrings_to_stub(module, context, combined_blacklist)
        elif any(path.is_relative_to(p) for p in package_paths):
            add_docstrings_to_stub(module, context, combined_blacklist)
    m = "\n--- Successfully codemodded typeshed ---"
    print(colored(m, "green"))
    sys.exit(0)


if __name__ == "__main__":
    main()
