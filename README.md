# Apertus on AMD

Run Megatron training on CSCS Alps with pinned source, published container images, direct Slurm launches, and reusable compiler caches.
Start with the included small dense checks before creating larger experiments.

## 1. Set up the repositories

You need [uv](https://docs.astral.sh/uv/getting-started/installation/).
The supplied site configuration targets Beverin's MI300A nodes.
Python 3.12 is managed by uv; PyTorch and ROCm come from the container.

```bash
git clone https://github.com/boeschf/apertus-amd.git
git clone https://github.com/boeschf/Megatron-LM-MoE.git
cd apertus-amd
uv sync --locked

export MEGATRON="$(cd ../Megatron-LM-MoE && pwd)"
git -C "$MEGATRON" checkout --detach c56351143a8e15899ef3d7d7f97867772f8e0e5c
cp config.toml local.toml
export SITE=local.toml
```

This repository provides orchestration; `Megatron-LM-MoE` provides the training code.
The examples select the commit recorded in their `experiment.toml`; the runner reads committed Git objects directly from `--checkout` and ignores uncommitted changes.
The examples exclude the unused `_research` submodule and do not require a recursive clone.
If you already have a checkout containing the selected commit, set `MEGATRON` to that path and reuse it; no clone or branch switch is needed.

## 2. Configure the site and image

`config.toml` is the checked-in site configuration; `local.toml` is the personal copy created by `cp config.toml local.toml` above.
`local.toml` is Git-ignored, so you can keep your account and storage paths there.
It is a complete configuration file, not an overlay: `--site "$SITE"` selects it instead of `config.toml`.
Without `--site`, commands read `config.toml` from the current directory.
Edit `local.toml` for your Slurm account/partition, shared `data_root`, image store, and machine topology.
The supplied topology is four GPUs/ranks per node, 192 logical CPUs, full allocatable memory, and one NUMA domain per rank.
Ranks retain visibility of the allocated GPUs and select their device using their Slurm local rank.
The CSCS site setting `uenv_passthrough = "ignore"` makes batch submission independent of a uenv loaded in the submitting shell.
The generated container EDF sets `entrypoint = true` and launches a small Bash `exec` wrapper so the image's `BASH_ENV` initializes its runtime defaults, including Slingshot/libfabric settings.
Use the site environment for intentional overrides.
The supplied site enables RCCL P2P explicitly and excludes Open MPI's memory patcher (`OMPI_MCA_memory=^patcher`) to avoid an HSA/libfabric lock inversion during IPC setup.

Images are published to a shared-filesystem registry accessible from the login and compute nodes.
Each image is stored as an immutable, SHA-256-addressed object alongside its manifest and recipe source archive, so users can reuse the same published image across experiments.
Each experiment selects its published image by SHA-256 in `experiment.toml`.
The supplied logical store maps to `/capstor/store/cscs/cscs/public/team-ml/apertus-amd-images` in the site configuration.
To select another published image, update `[image].store` and `[image].sha256` in the experiment and its store path in the site file.
The expected layout is `<store>/objects/sha256/<sha256>/{image.sqsh,manifest.json,source.tar.gz}`.
Images supply training dependencies; the selected Megatron source is staged separately at runtime.
The runner explicitly disables CSCS CXI/libfabric and AWS OFI injection so the image's built-in network libraries are used.
Every generated EDF sets `com.hooks.cxi.enabled = "false"` and `com.hooks.aws_ofi_nccl.enabled = "false"` under `[annotations]`, for all images.
It also sets `HWLOC_COMPONENTS = "-gl"` to prevent hwloc's OpenGL display discovery from blocking OFI initialization.
See [Build and publish images](#build-and-publish-images) to add an image to this registry.

## 3. Plan, then run an experiment

Choose a workload from the [experiment guide](experiments/EXPERIMENTS.md) and inspect its `experiment.toml`.
The following example starts with the small dense restart check.

Inspect an invocation before submitting it; planning writes nothing and starts no jobs:

```bash
uv run --locked apertus plan experiments/dense-baseline-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --submit --time 00:30:00

uv run --locked apertus run experiments/dense-baseline-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --submit --time 00:30:00 --checkout "$MEGATRON"
```

`run --submit` prints the run directory and submitted job ID, then returns.
Monitor the allocation with `squeue -j JOB_ID` and read the run's `result.json` after completion.
For an allocation you already own, replace `--submit` with `--job-id JOB_ID`; this invocation waits for the training step.
Both modes use the same frozen controller and direct-srun runtime.

After the one-node check, submit the multi-node experiment:

```bash
uv run --locked apertus run experiments/dense-multinode-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --submit --time 00:30:00 --checkout "$MEGATRON"
```

If you have reservation access, add `--reservation NAME` when submitting.
To queue a run after another job, use `--dependency afterany:JOB_ID`, or `afterok:JOB_ID` to require its success.
Resource overrides include `--nodes`, `--gpus-per-node`, `--cpus-per-task`, `--memory-gb`, `--time`, `--cpu-bind`, and `--distribution`.
Each included experiment records its intended allocation in top-level `nodes`; precedence is `--nodes`, then experiment `nodes`, then site `[slurm].nodes`.
Rank placement is block across nodes; the site supplies within-node placement, CPU binding, MPI, and network options.
Keep native batch sizes and parallelism arguments consistent with the selected GPU count.

## Results and cache reuse

### Diagnose distributed collectives

Before investigating a training communication failure, run the image CI's native RCCL all-reduce and GPU-aware MPI all-to-all checks on two nodes:

```bash
uv run --locked apertus run experiments/distributed-collectives-i0662f3a37ef3 \
  --site "$SITE" --submit --time 00:10:00
```

Add `--reservation NAME` when needed, or replace `--submit` with `--job-id JOB_ID` for an existing two-node allocation.
This workload uses image-provided executables and requires no Megatron checkout.
Commands are attempted in independent Slurm steps, each bounded by the experiment's timeout, with per-rank logs and executable hashes under `output/<command-index>/`.
`result.json` records all-rank command outcomes; inspect RCCL's numerical error count and complete size sweep separately from MPI timings.
Compiler caching and training comparisons do not apply to this workload.
Effective transport settings are retained in `output/environment-<rank>.json` for training and in each command's output directory for native tests.
`launch.json` records the host-side `SLURM_NETWORK` setting passed before `srun`.

### Training results

Runs live under `<data-root>/runs/<experiment>/<run>/`.
These run artifacts are the outcome records; experiment-specific guidance lives in [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md).
`resolved.json` records inputs, `launch.json` records the Slurm step, and `result.json` separates execution/check outcomes from cache publication.
Logs, checkpoints, per-rank records, and restart checks are in `output/`; batch jobs also write `output/slurm.log`.
Inspect `output/launcher.log` and trial `rank-*.log` files when a run fails.

### Compiler cache

The shared compiler cache reduces startup time by reusing compiled code across compatible runs and experiments.
It stores portable Triton kernel groups, compiled AITER modules, and source-matched Megatron dataset helpers under `<data-root>/cache/`.
Each rank restores these into a private cache in job-owned node-local scratch and compiles any missing specializations there.
After execution and any requested checks pass, the runner collects supported artifacts and atomically publishes a new immutable cache generation.
Mutable autotuning state and arbitrary build trees stay private.

Cache mode defaults to `auto`: restore a compatible generation if available, otherwise start cold.
`--cache-mode cold` bypasses restoration; `--cache-mode warm` requires an existing compatible generation.
New compiler keys are added to the cache, while existing complete objects win if independent builds differ.
Compatibility includes image, source, GPU target, and configured environment/preparation; different tensor sizes can add new specializations in the same namespace.

To compare cold and warm numerical results, run the same restart experiment twice, then pass the successful cold run directory to the warm invocation:

```bash
uv run --locked apertus run experiments/dense-baseline-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --job-id JOB_ID --checkout "$MEGATRON" --cache-mode cold
uv run --locked apertus run experiments/dense-baseline-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --job-id JOB_ID --checkout "$MEGATRON" --cache-mode warm \
  --compare-run /absolute/path/to/cold-run
```

The restart checker supports dense AdamW, unsharded Torch checkpoints, and TP/PP size one.
Scheduler completion alone does not establish numerical correctness.
After all users of a data root have stopped, its run directories and cache can be deleted to start fresh; shared source snapshots must remain while retained runs depend on them.
A cold run does not replace existing shared cache objects.

## Create an experiment

Copy an example directory and edit its `experiment.toml`.
Name it `<workload>[-variant]-m<source>-i<image>`, using the first 12 hexadecimal characters of the Megatron commit and published image SHA-256.
For image-native commands, omit `-m<source>`; choose a readable workload/variant label and keep the complete directory name within 64 lowercase letters, digits and hyphens.
When changing either pin, rename the experiment directory to match; full hashes in `experiment.toml` remain authoritative.
It owns the intended `nodes`, source commit, image selection, entry point, native Megatron `args`, environment, preparation commands, and procedure.
For ordinary training, replace the entire `[procedure]` table with `kind = "train"`.
This runs the selected entry point through the source-verification wrapper without smoke-test instrumentation or checkpoint comparisons.
Use `kind = "restart"` with an explicit `split` and `validation` script for restart validation, as in the dense examples.
For image-native tests, use `kind = "commands"` with explicit command argument arrays and a timeout in seconds, as in the collectives example.
For source changes, commit them in your Megatron checkout and update `[source].commit`; pushing is not required.
Site settings belong in the site file, and reusable execution logic belongs in `src/apertus_amd/`.

### Experiment validation

Workload-specific instrumentation and checks live under `experiments/validation/`, shared by experiments that need the same validation.
An optional `[procedure].validation` names a self-contained Python script relative to `experiment.toml`.
The runner copies it into the run directory and calls `validation.py train OUTPUT SOURCE ENTRYPOINT [ARGS...]` for each training phase, then `validation.py check RUN` after all ranks finish.
The training wrapper can call `apertus_amd.entrypoint.train(..., before_run=...)` to install its instrumentation while retaining the common source verification.
The check command writes `RUN/output/checks.json` and must exit nonzero on failure; cache publication follows successful checks.
For `--compare-run`, the validator also compares against `RUN/reference.json`.
The shared runtime does not interpret model classes, optimizer internals, or checkpoint formats.

## Build and publish images

The image recipes live in a separate repository:

```bash
git clone --branch apps/apertus2-megatron-rocm-imanol \
  https://github.com/boeschf/alps-extended-images.git ../alps-extended-images
./scripts/build.py --source ../alps-extended-images --list
```

Building requires a prepared environment/allocation with Podman, Enroot, and SquashFS tools, plus write access to the publication store.
Use native rootless Podman with subordinate UID/GID mappings and owned node-local storage; package installation needs to assign ownership to users inside the image.
The tools do not allocate compute resources or configure Podman storage.
Run the build commands on that prepared build node, from this repository's root.
Set `TMPDIR` to an owned node-local directory with room for the unpacked image; export staging there avoids slow small-file writes to shared storage.
The final image and evidence go to `--output`; `evidence/export-workspace.json` records the temporary workspace, which is removed after successful export verification and retained on failure.
The build script automatically works around the known Enroot Podman cleanup bug when it detects the affected helper, using a private library copy and recording the patch and hashes in the build evidence.
It discovers the helper location from the installed Enroot launcher or honors `ENROOT_LIBRARY_PATH` for a custom installation.
Actual export failures still fail the build, and publication requires a successful full filesystem-read check.

### Build a base, then its application image

Choose a new shared output directory for each build; existing output directories are rejected.
The example below builds the Primus 26.8 base and its Apertus2 application image.
Keep both builds on the same node and in the same Podman storage so the application build can find the locally tagged base.
Use the same `IMAGE_PREFIX` for both if you override the default `localhost/alps-images`.

```bash
export RECIPES="$(cd ../alps-extended-images && pwd)"
export BUILDS="$PWD/artifacts/my-image-builds"
export IMAGE_STORE=/path/to/writable/shared/image-store

./scripts/build.py --source "$RECIPES" \
  --output "$BUILDS/primus268-base" base rocm primus 26.8
./scripts/build.py --source "$RECIPES" \
  --output "$BUILDS/apertus268" app apertus-2 megatron-primus268-rocm
```

The build tool snapshots the working recipe context, builds from that snapshot, and verifies the exported image.
Each successful output contains `image.sqsh`, `source.tar.gz`, `provenance.json`, and logs in `evidence/`.
A failed build retains its workspace and logs.
For a standalone ROCm 10 base instead, use selector `base rocm pytorch rocm10.0-ubuntu24.04-py3.12-torch2.13` with its own output directory.

### Publish and select the image

Publication can run independently on a node with access to the build outputs and destination store:

```bash
PUBLISHED_IMAGE=$(./scripts/publish.py "$BUILDS/apertus268/image.sqsh" \
  "$BUILDS/apertus268/source.tar.gz" "$BUILDS/apertus268/provenance.json" \
  --store "$IMAGE_STORE")
printf '%s\n' "$PUBLISHED_IMAGE"
```

The publisher prints `<store>/objects/sha256/<sha256>/image.sqsh` and stores immutable checksum-addressed artifacts.
Set an entry in your site's `[image_stores]` table to `$IMAGE_STORE`, then set the experiment's `[image].store` to that entry's name and `[image].sha256` to the printed image's parent directory name.
To publish the base for collective tests, pass its three output files to the same publisher and select its checksum in the collective experiment.
Publication verifies artifact integrity; GPU and multi-node correctness require running the workloads described above.

### Combined build and publish

`scripts/build-publish.py` combines both stages in a temporary workspace, removes it after success, and retains it on failure:

```bash
./scripts/build-publish.py --source "$RECIPES" --store "$IMAGE_STORE" \
  base rocm pytorch rocm10.0-ubuntu24.04-py3.12-torch2.13
```

The scripts are independently copyable; the combined wrapper needs `build.py` and `publish.py` beside it.
Use each script's `--help` for build/export options.

## Development

Run `uv run --locked pytest -q` for CPU and scheduler-stub checks.
Contributor conventions are in [AGENTS.md](AGENTS.md).
