"""CPU-only execution of conditions extracted from two checksum-pinned routers.

No Megatron/Torch imports, tensors, tie ordering, gradients, or GPU execution are
covered here. Native GPU acceptance remains the separate EP16 component probe.
"""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from types import SimpleNamespace


BASE = "1072583a9632fd1902544f7eaea562e2578ef1de"
FIX = "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
ROUTER = "megatron/core/transformer/moe/router.py"
SOURCE_SHA256 = {
    BASE: "41a9f702a138176ee9e199210756a94c894161d80f20f168c448d8e432da64a6",
    FIX: "facf8a03dff9fba668f2da4d1e617174854aadad5e5475006622a379e2de6c3b",
}
EXCERPT_SHA256 = {
    BASE: "42673acc2748f392897a3331be40adae4fd2180605bbcab18c29c4f3f9570514",
    FIX: "780addb51ece46d7fbfa6b9fe81d0a5c81501e8388e3ba0f36708db7501853d9",
}
METHODS = ("histogram", "marin_histogram", "marin_exact", "average", "legacy_average")


def digest(data: bytes) -> str:
    """Return a byte-exact SHA-256 fingerprint."""
    return hashlib.sha256(data).hexdigest()


def one(nodes):
    """Reject missing or ambiguous native AST targets rather than guessing."""
    values = list(nodes)
    if len(values) != 1:
        raise ValueError(f"expected one native AST target, found {len(values)}")
    return values[0]


def assignment(tree, name):
    """Locate the unique simple assignment to a native local name."""
    return one(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == name)


def selector(tree):
    """Find the actual branch assigning topk_result, not a rewritten oracle."""
    return one(node for node in ast.walk(tree) if isinstance(node, ast.If)
               and any(isinstance(child, ast.Assign) and any(
                   isinstance(target, ast.Name) and target.id == "topk_result"
                   for target in child.targets) for child in node.body))


def statistics(tree):
    """Find the complete native beta-statistics block."""
    return one(node for node in ast.walk(tree) if isinstance(node, ast.If)
               and isinstance(node.test, ast.Name) and node.test.id == "should_update_beta")


def extract(source: str) -> tuple[str, list[dict]]:
    """Retain exact dedented native methods and metric conditions with line provenance."""
    tree = ast.parse(source)
    router = one(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TopKRouter")
    routing = one(node for node in router.body if isinstance(node, ast.FunctionDef) and node.name == "routing")
    nodes = [one(node for node in router.body if isinstance(node, ast.FunctionDef) and node.name == name)
             for name in ("quantile_balancing", "_apply_expert_bias")]
    nodes.extend(assignment(routing, name) for name in ("collect_training_metrics", "collect_inference_metrics"))
    fragments, ranges = [], []
    lines = source.splitlines(keepends=True)
    for node in nodes:
        raw = "".join(lines[node.lineno - 1:node.end_lineno])
        fragments.append(textwrap.dedent(raw).rstrip() + "\n")
        ranges.append({"first_line": node.lineno, "last_line": node.end_lineno,
                       "native_text_sha256": digest(raw.encode())})
    return "\n".join(fragments), ranges


def verify_sources(repository: Path) -> tuple[dict, dict]:
    """Read only the two committed router blobs; prove all other router AST is unchanged."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args])

    if git("rev-parse", FIX + "^").decode().strip() != BASE:
        raise ValueError("routing fix has an unexpected parent")
    if git("diff-tree", "--no-commit-id", "--name-only", "-r", FIX).decode().splitlines() != [ROUTER]:
        raise ValueError("routing commit changes files outside router.py")
    sources, trees, records = {}, {}, {}
    for commit, expected in SOURCE_SHA256.items():
        raw = git("show", f"{commit}:{ROUTER}")
        if digest(raw) != expected:
            raise ValueError(f"native source checksum mismatch: {commit}")
        sources[commit] = raw.decode()
        trees[commit] = ast.parse(sources[commit])
        excerpt, ranges = extract(sources[commit])
        if digest(excerpt.encode()) != EXCERPT_SHA256[commit]:
            raise ValueError(f"native excerpt extraction changed: {commit}")
        records[commit] = {"router_sha256": expected, "excerpt_sha256": digest(excerpt.encode()),
                           "blob": git("rev-parse", f"{commit}:{ROUTER}").decode().strip(),
                           "ranges": ranges}
    original, fixed = (trees[commit] for commit in (BASE, FIX))
    old_branch, new_branch = selector(original), selector(fixed)
    if ast.dump(old_branch.test) == ast.dump(new_branch.test):
        raise ValueError("expected selection-condition change is missing")
    original = deepcopy(original)
    selector(original).test = deepcopy(new_branch.test)
    if ast.dump(original) != ast.dump(fixed):
        raise ValueError("native router changes more than the selection condition")
    return sources, {"scope": "CPU AST/guard checks only; not native tensor or GPU validation",
                     "parent": BASE, "commit": FIX, "path": ROUTER, "sources": records,
                     "only_router_changed": True, "only_selection_test_ast_changed": True}


def load_excerpt(path: Path, commit: str):
    """Reject altered portable excerpts before compiling their native conditions."""
    raw = path.read_bytes()
    if commit not in EXCERPT_SHA256 or digest(raw) != EXCERPT_SHA256[commit]:
        raise ValueError("untrusted native excerpt")
    return ast.parse(raw.decode())


def conditions(tree, *, training: bool, grad_enabled: bool, method: str,
               capacity=None, drop_policy="probs", training_metrics="mbs", inference_metrics="mbs") -> dict:
    """Evaluate the retained native conditions with scalar stand-ins, never fake tensor routing."""
    config = SimpleNamespace(moe_router_quantile_balancing_method=method,
                             moe_expert_capacity_factor=capacity, moe_token_drop_policy=drop_policy,
                             moe_router_violation_metrics=training_metrics,
                             moe_router_inference_violation_metrics=inference_metrics)
    namespace = {"self": SimpleNamespace(training=training, config=config, topk=8),
                 "torch": SimpleNamespace(is_grad_enabled=lambda: grad_enabled)}
    # Execute the actual assignments before evaluating their dependent branch.
    for name in ("should_update_beta", "use_histogram", "use_marin", "use_exact", "compute_drop_priority",
                 "collect_training_metrics", "collect_inference_metrics"):
        exec(compile(ast.Module(body=[assignment(tree, name)], type_ignores=[]), "<native-assign>", "exec"), namespace)

    def evaluate(node):
        return eval(compile(ast.Expression(body=node), "<native-condition>", "eval"), namespace)

    branch = selector(tree)
    selected_body = branch.body if evaluate(branch.test) else branch.orelse
    call = one(node for statement in selected_body for node in ast.walk(statement)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "topk")
    bias = one(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_expert_bias")
    load_guard = one(node for node in bias.body if isinstance(node, ast.If))
    return {"topk_request": evaluate(call.args[0]),
            "statistics_update": bool(evaluate(statistics(tree).test)),
            "expert_load_update": bool(evaluate(load_guard.test)),
            "training_metric_update": bool(namespace["collect_training_metrics"]),
            "inference_metric_update": bool(namespace["collect_inference_metrics"])}


def main():
    """Capture/reverify committed evidence without importing Megatron or touching its worktree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new evidence directory; never overwrite retained evidence")
    args = parser.parse_args()
    sources, report = verify_sources(args.repository)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        for commit, source in sources.items():
            (args.output / f"router-{commit[:12]}.py.txt").write_text(extract(source)[0])
        (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
