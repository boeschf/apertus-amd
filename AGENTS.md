# Contributor conventions

## Structure and documentation

- README.md is the user guide; keep setup, repository URLs, examples, defaults, and supported workflows accurate and self-contained.
- Keep experiment-specific purpose and interpretation in `experiments/EXPERIMENTS.md`, with authoritative settings in each `experiment.toml` and outcomes in run artifacts.
  Do not turn README.md into an experiment catalog or run diary.
- Experiment directories contain explicit `experiment.toml` inputs; reusable orchestration belongs in `src/apertus_amd/`.
- Keep workload-specific instrumentation and numerical validators in `experiments/validation/`, selected explicitly by experiments.
- Keep machine paths, scheduler defaults, and topology in site configuration.
  Keep intended node counts, native training arguments, workload environment, and preparation commands explicit in experiments.
- Image recipes belong in their separate repository.
  Keep `scripts/build.py` and `scripts/publish.py` independently copyable and standard-library-only; `build-publish.py` is their thin adjacent wrapper.
- Write Markdown with one sentence per line, preserving tables and code blocks.

## Development

- Use `uv sync --locked`, `uv run --locked`, and `uv add`/`uv remove` for environments, project commands, and dependency changes.
- Prefer straightforward functions and clear names; add abstractions for concrete responsibilities or demonstrated duplication.
- Give public modules and functions concise contract docstrings.
  Comment non-obvious ordering, filesystem assumptions, concurrency, ownership, and failure behavior.
- Keep shell entry points small; never add strict shell options to fragments sourced through `BASH_ENV`.
- Preserve user work and external repositories/images unless changes are explicitly authorized.

## Execution contracts

- Disable both CXI/libfabric and AWS OFI injection in every EDF; images provide the complete pinned network stack.
- Set `HWLOC_COMPONENTS=-gl` in every EDF to avoid blocking OpenGL display discovery during OFI initialization.
- Plan without subprocesses or writes; use the same resolver and direct-srun runtime for existing allocations and batch jobs.
- Freeze committed source and run tooling, verify executed source imports, and retain authoritative resolved inputs and results.
- Keep allocated GPUs visible and preserve Slurm local-rank device selection; rank placement across nodes is block.
- Use job-owned scratch and private rank caches, bounded coordination, and cleanup only after local children exit.
- Capture quiescent caches and publish only after requested checks pass.
  Verify integrity, keep existing whole objects, add missing keys, and publish immutable generations atomically under a bounded lock.
- Keep image publication independent of experiment execution; normal runs trust immutable published image identity without rehashing the image.

## Verification

- Run focused checks for each change, including failure paths for lifecycle changes; use `uv run --locked pytest -q` for the complete CPU suite.
- Use scheduler stubs for submission tests and explicitly authorized allocations for real GPU/multi-node checks.
- Use workload-appropriate numerical checks and keep baseline training independent of reference optimizations.
- Distinguish scheduler completion, finite updates, checkpoint integrity, numerical equivalence, and cache reuse.
  CPU tests and dry runs do not establish GPU correctness.
