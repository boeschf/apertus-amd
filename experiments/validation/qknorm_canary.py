"""Canary wrapper; use after the frozen B1 memory validator has been loaded."""

import sys


def qk_canary_run():
    operation, root = sys.argv[1:3]
    if operation == "train":
        output, source, entrypoint, *args = sys.argv[2:]
        output = Path(output)
        # The canonical runner supplies a checkpoint path; a 10-update smoke
        # does not need to write a multi-terabyte model/optimizer checkpoint.
        if args.count("--save") != 1:
            raise ValueError("expected one canonical save argument")
        index = args.index("--save")
        if args[index + 1] != str(output / "checkpoints"):
            raise ValueError("unexpected checkpoint destination")
        del args[index:index + 2]
        if os.environ.get("BEVERIN_QKNORM_FLA") != "1" or os.environ.get("UCCL_EP_TRANSPORT") != "cxi":
            raise ValueError("missing canary selectors")
        trace_training(output, Path(source), entrypoint, args)
        # Read only already-loaded autotuners after training.  NB is ceil(T/2048)
        # in both pinned FLA kernels: 56 for Q and 14 for K, with D=128.
        norm = sys.modules.get("fla.modules.layernorm")
        evidence = {}
        for phase, name in (("forward", "layer_norm_fwd_kernel"), ("backward", "layer_norm_bwd_kernel")):
            runner = getattr(norm, name, None)
            while runner is not None and not isinstance(getattr(runner, "cache", None), dict):
                runner = getattr(runner, "fn", None)
            keys = getattr(runner, "cache", {})
            evidence[phase] = sorted({k[1] for k in keys if type(k) is tuple and len(k) >= 5 and
                                      k[0] == 128 and k[1] in (14, 56) and
                                      k[2:5] == (False, False, True) and "torch.bfloat16" in k[5:]})
        write_json(output / f"qknorm-route-{os.environ['RANK']}.json", {
            "rank": int(os.environ["RANK"]), "evidence": "post-run FLA RMSNorm autotuner cache keys",
            "nb": evidence, "checkpoint_requested": False})
    elif operation == "check":
        root = Path(root)
        check(root)
        baseline = json.loads((root / "output/checks.json").read_text())
        routes = [json.loads((root / f"output/qknorm-route-{r['rank']}.json").read_text())
                  for r in baseline["ranks"]]
        covered = {phase: sorted({key for rec in routes for key in rec["nb"][phase]})
                   for phase in ("forward", "backward")}
        baseline["qknorm"] = {"covered_nb": covered, "ranks_with_both_sizes": [
            r["rank"] for r in routes if r["nb"] == {"forward": [14, 56], "backward": [14, 56]}],
            "scope": "canary; not per-call tracing or numerical qualification"}
        write_json(root / "output/checks.json", baseline)
        if covered != {"forward": [14, 56], "backward": [14, 56]}:
            raise ValueError("training exited but Q/K FLA route evidence is incomplete")
    else:
        raise ValueError(f"unknown canary operation: {operation}")


if __name__ == "__main__":
    qk_canary_run()
