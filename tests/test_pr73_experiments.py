"""Keep the P2P-restoration experiment explicit and independent of site defaults."""

from pathlib import Path

from apertus_amd.config import resolve
from apertus_amd.runs import edf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/collectives-pr73-icb5a384016a9"


def test_pr73_collectives_enable_p2p_and_validate_all_tested_paths():
    value = resolve(EXPERIMENT, submit=True)
    assert value["resources"]["nodes"] == 2
    assert value["image"]["sha256"].startswith("cb5a384016a9")
    expected = {
        "NCCL_P2P_DISABLE": "0", "OFI_NCCL_PROTOCOL": "SENDRECV", "NCCL_CROSS_NIC": "0",
        "NCCL_PXN_DISABLE": "1", "NCCL_NET_GDR_LEVEL": "PHB", "NCCL_NET_FORCE_FLUSH": "1",
        "HSA_ENABLE_IPC_MODE_LEGACY": "1", "NCCL_P2P_LEVEL": "SYS", "OMPI_MCA_memory": "^patcher",
    }
    assert value["environment"].items() >= expected.items()
    commands = value["procedure"]["commands"]
    assert [command[0] for command in commands] == [
        "all_reduce_perf", "alltoall_perf", "alltoallv_perf", "sendrecv_perf",
    ]
    assert all(command[-2:] == ["-c", "1"] for command in commands)
    text = edf(value, Path("/run"))
    assert 'com.hooks.cxi.enabled = "false"' in text
    assert 'com.hooks.aws_ofi_nccl.enabled = "false"' in text
    assert 'HWLOC_COMPONENTS = "-gl"' in text


def test_pr73_intranode_arm_uses_the_same_settings():
    intra = resolve(EXPERIMENT, submit=True, nodes=1)
    inter = resolve(EXPERIMENT, submit=True)
    assert intra["resources"]["nodes"] == 1
    assert intra["environment"] == inter["environment"]
    assert intra["procedure"] == inter["procedure"]


def test_pr73_chonk_changes_only_network_environment():
    baseline = resolve(ROOT / "experiments/moe-chonk-kda-m1072583a9632-icb5a384016a9", submit=True)
    candidate = resolve(ROOT / "experiments/moe-chonk-kda-pr73-m1072583a9632-icb5a384016a9", submit=True)
    for key in ("args", "source", "image", "prepare", "procedure", "resources"):
        assert candidate[key] == baseline[key]
    changed = {key for key in candidate["environment"]
               if candidate["environment"][key] != baseline["environment"].get(key)}
    assert changed == {
        "NCCL_P2P_DISABLE", "OFI_NCCL_PROTOCOL", "NCCL_CROSS_NIC", "NCCL_PXN_DISABLE",
        "NCCL_NET_GDR_LEVEL", "NCCL_NET_FORCE_FLUSH", "HSA_ENABLE_IPC_MODE_LEGACY", "NCCL_P2P_LEVEL",
    }


def test_pr73_recompute_diagnostic_preserves_network_and_model():
    baseline = resolve(ROOT / "experiments/moe-chonk-kda-pr73-m1072583a9632-icb5a384016a9", submit=True)
    candidate = resolve(ROOT / "experiments/moe-chonk-kda-pr73act-m1072583a9632-icb5a384016a9", submit=True)
    for key in ("source", "image", "prepare", "procedure", "resources", "environment"):
        assert candidate[key] == baseline[key]
    expected = baseline["args"].copy()
    expected[expected.index("moe")] = "moe_act"
    for flag in ("--train-iters", "--save-interval"):
        expected[expected.index(flag) + 1] = "26"
    assert candidate["args"] == expected


def test_pr73act_matched_p2p_control_changes_only_one_environment_key():
    baseline = resolve(ROOT / "experiments/moe-chonk-kda-pr73act-m1072583a9632-icb5a384016a9", submit=True)
    candidate = resolve(ROOT / "experiments/moe-chonk-kda-pr73act-nop2p-m1072583a9632-icb5a384016a9", submit=True)
    for key in ("args", "source", "image", "prepare", "procedure", "resources"):
        assert candidate[key] == baseline[key]
    assert candidate["environment"] == {**baseline["environment"], "NCCL_P2P_DISABLE": "1"}


def test_pr73_eager_control_changes_compile_and_documented_duration_only():
    baseline = resolve(ROOT / "experiments/moe-chonk-kda-pr73-m1072583a9632-icb5a384016a9", submit=True)
    candidate = resolve(ROOT / "experiments/moe-chonk-kda-pr73-eager-m1072583a9632-icb5a384016a9", submit=True)
    for key in ("source", "image", "prepare", "procedure", "resources"):
        assert candidate[key] == baseline[key]
    expected = baseline["args"].copy()
    for flag in ("--train-iters", "--save-interval"):
        expected[expected.index(flag) + 1] = "26"
    assert candidate["args"] == expected
    assert candidate["environment"] == {**baseline["environment"], "TORCH_COMPILE_DISABLE": "1"}
