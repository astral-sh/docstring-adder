"""Script that attempts to invoke docstring-adder on all third-party typeshed stubs."""

import argparse
import subprocess
import tempfile
from pathlib import Path


def add_docstrings(typeshed_dir: Path, docstring_adder_dir: Path) -> None:
    """Add docstrings to third-party typeshed stubs."""

    packages = list((typeshed_dir / "stubs").iterdir())
    for i, path in enumerate((typeshed_dir / "stubs").iterdir(), start=1):
        print(f"\nCodemodding package [{i}/{len(packages)}]\n")
        with tempfile.TemporaryDirectory() as td:
            venv_dir = f"{td}-venv"
            subprocess.run(["uv", "venv", "--python", "3.13", venv_dir], check=True)

            # seems to fail with an internal assertion when you try to import it...?
            if path.name == "keyboard":
                continue

            try:
                subprocess.run(
                    [
                        "uvx",
                        "--python",
                        f"{venv_dir}/bin/python",
                        "--from",
                        docstring_adder_dir,
                        "add-docstrings",
                        "--typeshed-packages",
                        path,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                if e.returncode == 10:
                    print(f"\nFailed to install runtime package for `{path.name}\n`")
                    continue
                else:
                    raise


if __name__ == "__main__":

    def absolute_path(path: str) -> Path:
        """Convert a string path to an absolute Path."""
        return Path(path).absolute()

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--typeshed-dir",
        type=absolute_path,
        help="Path to the typeshed directory.",
        required=True,
    )
    parser.add_argument(
        "--docstring-adder-dir",
        type=absolute_path,
        help="Path to docstring-adder.",
        required=True,
    )
    args = parser.parse_args()
    add_docstrings(
        typeshed_dir=args.typeshed_dir, docstring_adder_dir=args.docstring_adder_dir
    )
