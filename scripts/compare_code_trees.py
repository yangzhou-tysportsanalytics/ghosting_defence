"""Compare two Python source trees file by file, ignoring comments and docstrings.

For each file: whether the text is identical, whether the code is identical after removing
docstrings (comments never reach the syntax tree), and the sha256 of that normalised code, so
that one side can be checked by anyone who has only the other side. ``--drop file:function``
also compares a file with the named top-level functions removed (additions on one side).

Usage:
    uv run python scripts/compare_code_trees.py --a <tree A> --b <tree B> --files f1.py f2.py ...
        [--drop xfg/api.py:read_release_shots] [--out result.json] [--label-a ...] [--label-b ...]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def normalised(path: Path, drop: tuple[str, ...] = ()) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in drop)]
    for n in ast.walk(tree):
        if (
            isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(getattr(n.body[0], "value", None), ast.Constant)
            and isinstance(n.body[0].value.value, str)
        ):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compare(a: Path, b: Path, files: list[str], drops: dict[str, tuple[str, ...]]) -> dict:
    out = {}
    for f in files:
        pa, pb = a / f, b / f
        na, nb = normalised(pa), normalised(pb)
        row = {
            "text_identical": pa.read_text(encoding="utf-8") == pb.read_text(encoding="utf-8"),
            "code_identical": na == nb,
            "code_sha256_a": sha(na),
            "code_sha256_b": sha(nb),
        }
        if f in drops:
            row["code_identical_without"] = {
                "functions": list(drops[f]),
                "identical": normalised(pa, drops[f]) == normalised(pb, drops[f]),
            }
        out[f] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--drop", nargs="*", default=[], help="file:function (added on one side)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    drops: dict[str, tuple[str, ...]] = {}
    for d in args.drop:
        f, fn = d.split(":")
        drops[f] = (*drops.get(f, ()), fn)
    res = {"a": args.label_a, "b": args.label_b,
           "method": "Python AST with docstrings removed (comments are not in the AST)",
           "files": compare(Path(args.a), Path(args.b), args.files, drops)}  # fmt: skip
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
