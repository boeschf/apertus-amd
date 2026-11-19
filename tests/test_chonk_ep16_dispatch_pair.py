"""Check the matched full-model dispatcher experiment without scheduler or vendor imports."""

from pathlib import Path
import tomllib

from apertus_amd import config, runs


ROOT = Path(__file__).resolve().parents[1]


def test_ep16_training_pair_changes_only_dispatcher():
    """Reject changes to the workload, environment or resource policy between arms."""
    options = dict(site=ROOT / "config.toml", submit=True, reservation="cscs",
                   time="01:15:00", cache_mode="cold")
    a, b = [config.resolve(ROOT / "experiments" / f"moe-chonk-kda-ep16-{arm}-i910bfcf936d3", **options)
            for arm in ("alltoall", "uccl")]
    assert a.keys() == b.keys()
    assert {key for key in a if a[key] != b[key]} == {"experiment", "args"}
    index = a["args"].index("--moe-token-dispatcher-type") + 1
    assert (a["args"][index], b["args"][index]) == ("alltoall", "flex")
    assert b["args"] == a["args"][:index] + ["flex"] + a["args"][index + 1:]
    for key, value in (("--moe-flex-dispatcher-backend", "deepep"),
                       ("--expert-model-parallel-size", "16"), ("--train-iters", "60"),
                       ("--save-interval", "60"), ("--global-batch-size", "512")):
        assert config.option(a["args"], key) == value
    pos = a["args"].index("--recompute-modules")
    assert a["args"][pos + 1:pos + 4] == ["layernorm", "linear_attn", "moe_act"]
    assert (a["resources"]["nodes"], a["resources"]["gpus_per_node"]) == (16, 4)
    assert a["cache"]["mode"] == "cold"
    assert a["source"]["commit"] == "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
    assert a["image"]["sha256"] == "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
    edf = tomllib.loads(runs.edf(b, Path("/preview")))
    assert edf["entrypoint"] is True
    assert edf["env"] | {"UCCL_EP_TRANSPORT": "cxi", "UCCL_SOCKET_IFNAME": "hsn",
                         "UCCL_CXI_THREADING": "safe"} == edf["env"]
    assert not {"FI_CXI_ENABLE_WRITEDATA", "UCCL_EP_DISPATCH_CONFIG", "UCCL_EP_COMBINE_CONFIG"} & edf["env"].keys()
