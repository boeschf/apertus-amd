# Experiment guide

Each linked `experiment.toml` is the authoritative input: source commit, image, node count, native arguments, environment and procedure.
This guide explains purpose and interpretation; the root [README](../README.md) covers setup, execution and cache management.
Run commands below from the repository root, with `SITE` and the appropriate checkout configured as described there.

## Dense restart checks

| Experiment | Purpose |
|---|---|
| [0001-dense-baseline](0001-dense-baseline/experiment.toml) | Small one-node training and exact checkpoint/restart check |
| [0002-dense-multinode](0002-dense-multinode/experiment.toml) | Two-node version of the dense restart check |

Both use BF16, mock data, ordinary dense attention and AdamW, with continuous/prefix/resumed trials.
Both explicitly select [validation/dense_restart.py](validation/dense_restart.py), which owns their instrumentation and numerical checks.
They are correctness smoke tests, not convergence or performance benchmarks.

## Distributed collectives

[0003-distributed-collectives](0003-distributed-collectives/experiment.toml) runs the image CI's RCCL all-reduce and GPU-aware MPI all-to-all commands on two nodes.
It uses image-provided executables without a Megatron checkout, with a separate 180-second timeout per command.
RCCL output is under `output/0/` and MPI output under `output/1/`.
Inspect RCCL's numerical error count and complete size sweep separately from command exits and MPI timings.

## Mid-4B MoE baseline

[0004-moe-mid4b](0004-moe-mid4b/experiment.toml) selects prepared SwissAI Megatron source and the published Primus 26.8 Apertus application image.
It uses full Mid-4B geometry: 32 layers, hidden size 2560, 256 experts with top-8 routing, sequence length 8192, and a repeating three-KDA/one-global-attention pattern.
The first layer is dense; the remaining 31 are MoE.
The initial run uses mock data, the SwissAI magnitude/direction-decoupled optimizer, and 26 updates with a final checkpoint.
It retains the 718-update learning-rate warmup, so this run stays within warmup.

The experiment defaults to 16 nodes / 64 GPUs, using TP1/PP2/CP1/EP8/ETP1 and global batch 1152: DP32 with 36 microbatches per update.
Four nodes is the minimum rank count for this layout, not an established memory-capacity result.
Reuse a checkout containing the pinned commit:

```bash
uv run --locked apertus plan experiments/0004-moe-mid4b \
  --site "$SITE" --submit
uv run --locked apertus run experiments/0004-moe-mid4b \
  --site "$SITE" --submit --checkout "$MEGATRON"
```

This uses ordinary `[procedure] kind = "train"`; it has no numerical or restart validator.
Configuration checks do not establish full-shape GPU correctness, convergence, or throughput.

## Early communicator warmup comparison

[0005-moe-mid4b-comm-warmup](0005-moe-mid4b-comm-warmup/experiment.toml) selects the fork's backport of Imanol's communicator warmup onto the baseline SwissAI revision.
It retains the baseline model, batch, 26-update protocol and image, and enables early collective/pipeline/expert connections with `BEVERIN_COMM_WARMUP=1`, `BEVERIN_COMM_WARMUP_PARTS=ar,p2p,a2a` and `NCCL_RUNTIME_CONNECT=0`.
The module's synthetic memory-pressure test is explicitly disabled with `SS_BALLAST_GB=0`.
This is a communication-initialization diagnostic; successful warmup alone does not establish training or checkpoint correctness.
Use the prepared worktree containing the pinned commit:

```bash
uv run --locked apertus run experiments/0005-moe-mid4b-comm-warmup \
  --site "$SITE" --submit --reservation cscs --checkout ../megatron/comm-warmup
```

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
