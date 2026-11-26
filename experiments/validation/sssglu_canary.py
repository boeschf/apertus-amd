"""SSSGLU training canary entry point appended to the frozen B1 memory validator."""

import sys


def sssglu_routes(output, source):
    """Read live Triton cache inventories without importing or launching the helper."""
    helper = sys.modules.get("megatron.core.fusions.beverin_sssglu")
    entries = {"forward": 0, "backward": 0}
    origin = None
    if helper is not None:
        origin = Path(helper.__file__).resolve()
        if origin != source.resolve() / "megatron/core/fusions/beverin_sssglu.py":
            raise ValueError("SSSGLU helper imported outside the committed source")
        for phase, symbol in (("forward", "_fwd"), ("backward", "_bwd")):
            caches = vars(getattr(helper, symbol)).get("device_caches")
            if caches is None:
                raise ValueError("missing live Triton device caches")
            for state in caches.values():
                if not isinstance(state, tuple) or not state or not hasattr(state[0], "keys"):
                    raise ValueError("unexpected live Triton cache ABI")
                entries[phase] += len(state[0])
    torch = sys.modules["torch"]
    write_json(output / f"sssglu-route-{os.environ['RANK']}.json", {
        "rank": int(os.environ["RANK"]), "helper": str(origin) if origin else None,
        "entries": entries, "checkpoint_requested": False,
        "recompile_limit": torch._dynamo.config.recompile_limit,
        "accumulated_recompile_limit": torch._dynamo.config.accumulated_recompile_limit,
    })


def sssglu_canary_run():
    """Keep B1's telemetry and compiler policy; omit checkpoint writes for this canary."""
    operation, root = sys.argv[1:3]
    if operation == "train":
        output, source, entrypoint, *args = sys.argv[2:]
        output, source = Path(output), Path(source)
        if args.count("--save") != 1:
            raise ValueError("expected one canonical save argument")
        index = args.index("--save")
        if args[index + 1] != str(output / "checkpoints"):
            raise ValueError("unexpected checkpoint destination")
        del args[index:index + 2]
        if (os.environ.get("BEVERIN_FUSED_SSSGLU") != "1"
                or os.environ.get("BEVERIN_QKNORM_FLA", "0") not in ("", "0")
                or os.environ.get("UCCL_EP_TRANSPORT") != "cxi"):
            raise ValueError("unexpected canary selectors")
        trace_training(output, source, entrypoint, args)
        sssglu_routes(output, source)
    elif operation == "check":
        root = Path(root)
        check(root)
        output = root / "output"
        baseline = json.loads((output / "checks.json").read_text())
        routes = [json.loads((output / f"sssglu-route-{r['rank']}.json").read_text())
                  for r in baseline["ranks"]]
        observed = [r["rank"] for r in routes if all(n > 0 for n in r["entries"].values())]
        fixed_policy = all((r["recompile_limit"], r["accumulated_recompile_limit"]) == (8, 256)
                           for r in routes)
        baseline["sssglu"] = {
            "ranks_with_helper_cache_entries_forward_and_backward": observed,
            "default_compiler_budget": fixed_policy,
            "scope": "live Triton cache evidence; not per-call counts or numerical qualification",
        }
        write_json(output / "checks.json", baseline)
        if not fixed_policy or not observed:
            raise ValueError("SSSGLU compiler policy or live route evidence is incomplete")
    else:
        raise ValueError(f"unknown canary operation: {operation}")


if __name__ == "__main__":
    sssglu_canary_run()
