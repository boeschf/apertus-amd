# Experiment guide

Each linked `experiment.toml` is the authoritative input: source commit, image, node count, native arguments, environment and procedure.
This guide explains purpose and interpretation; the root [README](../README.md) covers setup, execution and cache management.
Run commands below from the repository root, with `SITE` and the appropriate checkout configured as described there.
Directory names combine a readable workload/variant label with `-m<12-character Megatron commit>-i<12-character image SHA-256>`; image-native workloads omit the Megatron suffix.
Existing run artifacts retain the experiment names recorded when they were created, including the earlier numbered names.

## Dense restart checks

| Experiment | Purpose |
|---|---|
| [dense-baseline-mc56351143a8e-ie84d7ba62309](dense-baseline-mc56351143a8e-ie84d7ba62309/experiment.toml) | Small one-node training and exact checkpoint/restart check |
| [dense-multinode-mc56351143a8e-ie84d7ba62309](dense-multinode-mc56351143a8e-ie84d7ba62309/experiment.toml) | Two-node version of the dense restart check |

Both use BF16, mock data, ordinary dense attention and AdamW, with continuous/prefix/resumed trials.
Both explicitly select [validation/dense_restart.py](validation/dense_restart.py), which owns their instrumentation and numerical checks.
They are correctness smoke tests, not convergence or performance benchmarks.

## Distributed collectives

[distributed-collectives-i0662f3a37ef3](distributed-collectives-i0662f3a37ef3/experiment.toml) runs the image CI's RCCL all-reduce and GPU-aware MPI all-to-all commands on two nodes.
It uses image-provided executables without a Megatron checkout, with a separate 180-second timeout per command.
RCCL output is under `output/0/` and MPI output under `output/1/`.
Inspect RCCL's numerical error count and complete size sweep separately from command exits and MPI timings.

## Mid-4B MoE baseline

[moe-mid4b-mc56351143a8e-ie84d7ba62309](moe-mid4b-mc56351143a8e-ie84d7ba62309/experiment.toml) selects prepared SwissAI Megatron source and the published Primus 26.8 Apertus application image.
It uses full Mid-4B geometry: 32 layers, hidden size 2560, 256 experts with top-8 routing, sequence length 8192, and a repeating three-KDA/one-global-attention pattern.
The first layer is dense; the remaining 31 are MoE.
The initial run uses mock data, the SwissAI magnitude/direction-decoupled optimizer, and 26 updates with a final checkpoint.
It retains the 718-update learning-rate warmup, so this run stays within warmup.

The experiment defaults to 16 nodes / 64 GPUs, using TP1/PP2/CP1/EP8/ETP1 and global batch 1152: DP32 with 36 microbatches per update.
Four nodes is the minimum rank count for this layout, not an established memory-capacity result.
Reuse a checkout containing the pinned commit:

```bash
uv run --locked apertus plan experiments/moe-mid4b-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --submit
uv run --locked apertus run experiments/moe-mid4b-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --submit --checkout "$MEGATRON"
```

This uses ordinary `[procedure] kind = "train"`; it has no numerical or restart validator.
Configuration checks do not establish full-shape GPU correctness, convergence, or throughput.

## Early communicator warmup comparison

[moe-mid4b-comm-warmup-m1072583a9632-ie84d7ba62309](moe-mid4b-comm-warmup-m1072583a9632-ie84d7ba62309/experiment.toml) selects the fork's backport of Imanol's communicator warmup onto the baseline SwissAI revision.
It retains the baseline model, batch, 26-update protocol and image, and enables early collective/pipeline/expert connections with `BEVERIN_COMM_WARMUP=1`, `BEVERIN_COMM_WARMUP_PARTS=ar,p2p,a2a` and `NCCL_RUNTIME_CONNECT=0`.
The module's synthetic memory-pressure test is explicitly disabled with `SS_BALLAST_GB=0`.
This is a communication-initialization diagnostic; successful warmup alone does not establish training or checkpoint correctness.
Use the prepared worktree containing the pinned commit:

```bash
uv run --locked apertus run experiments/moe-mid4b-comm-warmup-m1072583a9632-ie84d7ba62309 \
  --site "$SITE" --submit --reservation cscs --checkout ../megatron/comm-warmup
```

## Rebuilt Primus 26.8 Apertus image

These experiments select the rebuilt `megatron-primus268-rocm` application image `cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0` from the shared `apertus-amd-images` store.
They retain the corresponding workload and source settings above, with the dense checks on baseline Megatron and Mid-4B on the communicator-warmup backport.
The image makes reference optimizations opt-in, so these experiments omit the old image's `B2_AUTOTUNE_SKIP=0` startup-hook workaround.
The collective checks deliberately use the same Apertus application image as training.

| Experiment | Nodes | Procedure |
|---|---|---|
| [dense-baseline-mc56351143a8e-icb5a384016a9](dense-baseline-mc56351143a8e-icb5a384016a9/experiment.toml) | 1 | Dense checkpoint/restart check |
| [dense-multinode-mc56351143a8e-icb5a384016a9](dense-multinode-mc56351143a8e-icb5a384016a9/experiment.toml) | 2 | Distributed dense checkpoint/restart check |
| [distributed-collectives-icb5a384016a9](distributed-collectives-icb5a384016a9/experiment.toml) | 2 | RCCL all-reduce and GPU-aware MPI all-to-all |
| [moe-mid4b-comm-warmup-m1072583a9632-icb5a384016a9](moe-mid4b-comm-warmup-m1072583a9632-icb5a384016a9/experiment.toml) | 16 | Full-shape Mid-4B training with communicator warmup |

```bash
uv run --locked apertus run experiments/moe-mid4b-comm-warmup-m1072583a9632-icb5a384016a9 \
  --site "$SITE" --submit --reservation cscs --checkout ../megatron/comm-warmup
```

Image build/publication checks establish artifact integrity; use these experiments to assess GPU execution, communication and workload-specific correctness.

## Mid-4B allocator and compilation comparison

These three 16-node variants retain the communicator-warmup source, rebuilt image, and 26-update workload.
They make the two historical memory-related defaults explicit to separate allocator behavior from compilation.

| Experiment | Expandable segments | Disable `torch.compile` |
|---|---|---|
| [moe-mid4b-mem-both-m1072583a9632-icb5a384016a9](moe-mid4b-mem-both-m1072583a9632-icb5a384016a9/experiment.toml) | True | 1 |
| [moe-mid4b-mem-alloc-m1072583a9632-icb5a384016a9](moe-mid4b-mem-alloc-m1072583a9632-icb5a384016a9/experiment.toml) | True | 0 |
| [moe-mid4b-mem-eager-m1072583a9632-icb5a384016a9](moe-mid4b-mem-eager-m1072583a9632-icb5a384016a9/experiment.toml) | False | 1 |

Use `--cache-mode cold` for each comparison arm.
The supplied site configuration enables RCCL P2P with `NCCL_P2P_DISABLE=0` and excludes Open MPI's memory patcher with `OMPI_MCA_memory=^patcher` to avoid its HSA/libfabric lock inversion during IPC setup.
All three arms use these communication settings, which also differ from the earlier P2P-disabled historical runs.
All select [validation/memory_trace.py](validation/memory_trace.py), which samples process memory, available cgroup counters, GPU allocator counters, and node/shared-memory usage every ten seconds, with private-cache sizes every minute.
It preserves ordinary verified source execution and does not synchronize GPU work or reset memory peaks.
The `memory-start-<rank>.json`, `memory-<rank>.jsonl`, and `memory-end-<rank>.json` artifacts retain effective settings and memory evidence, including partial traces after a killed process.
Its final check verifies telemetry completeness and configured settings; it does not validate numerical equivalence or checkpoint contents.
The prior uninstrumented new-image run is a historical comparison, not a simultaneously measured fourth arm.

## Outcomes and evidence

Each invocation freezes its inputs and records its outcome under `<data-root>/runs/<experiment>/<run>/`.

| Artifact | Meaning |
|---|---|
| `resolved.json` | Exact executed inputs, source/image identity and resource settings |
| `submission.json` | Batch submission receipt and job ID |
| `launch.json` | Actual Slurm launch and controller outcome |
| `result.json` | Execution/check outcomes and cache publication status |
| `output/checks.json` | Numerical/checkpoint validation when an explicit validator provides it |
| `output/rank-*.log`, `output/launcher.log`, `output/slurm.log` | Training, launcher and batch evidence |

Use these records for results rather than maintaining a second pass/fail history in documentation.
Scheduler completion, completed updates, validated checkpoints and numerical equivalence are distinct claims.
