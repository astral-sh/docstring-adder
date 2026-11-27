<div align=center>

# docstring-adder

<br>

<img width="512" height="512" alt="A professorial adder inscribing documentation at his desk, on which lies a ball of string" src="https://github.com/user-attachments/assets/0eb96058-c110-4759-ade3-223f5734d20a" />

##

### A codemod to auto-add docstrings to stub files

</div>

---

It's easiest to use this tool using [uv](https://docs.astral.sh/uv/).

Either install the tool using `uv tool install git+https://github.com/astral-sh/docstring-adder`
and then invoke it using the `add-docstrings` command, or invoke it directly using
`uvx --from=git+https://github.com/astral-sh/docstring-adder add-docstrings`.

Consult the module-level docstring in `add_docstrings.py` for more details on how the tool works.
Run `uvx --from=git+https://github.com/astral-sh/docstring-adder add-docstrings --help` for
information on the various CLI options the tool supports.

## Module overrides

By default the tool imports the runtime module whose name matches each stub's own dotted
module name. When a stub's symbols actually live in a different runtime module -- for
example, extension modules that flatten many logical submodules into a single
compiled module -- you can redirect the lookup with a `[tool.docstring-adder.module-overrides]`
table:

```toml
[tool.docstring-adder.module-overrides]
"example_package.submodule" = "example_package._runtime"
"example_package.other_submodule" = "example_package._runtime"
```

Each entry maps a stub's dotted module name to the runtime module whose docstrings should
be used instead. The table is auto-discovered from a nearby `pyproject.toml`. The stub's
own module name is still used for blacklist matching and logging.
