#!/bin/bash
# Export rocpd databases to pftrace files using the training container image.
#
# Usage:
#   scripts/export-pftraces.sh <run-output-root> [extra sbatch options...]
#
#   <run-output-root>  the run's output directory, i.e.
#                      .../<experiment>/<run-id>/output
#                      Every */*_results.db under <run-output-root>/rocprof
#                      is exported to <node-dir>/<pid>.pftrace, next to its
#                      database.
#
# Example:
#   scripts/export-pftraces.sh \
#     artifacts/experiment-data/runs/moe-chonk-kda-ep16-uccl-rocprof-i910bfcf936d3/20261123T113609Z-d3968c8d/output
#
# What it does:
#   1. Locates the image used for the run (resolved.json, fallback container.toml).
#   2. Writes an EDF (Environment Definition File) mounting the run's output
#      directory read-write at /apertus/output, mirroring the run's own
#      container.toml (same mechanism the training job used).
#   3. Submits a single-node Slurm job that runs the image's own
#      /opt/venv/bin/rocpd2pftrace on every database inside the container.
#
#   Conversion must run inside the image: the image's system python has a
#   shared _sqlite3 module, which rocpd's gotcha-based sqlite interop
#   requires (the login-node python's _sqlite3 is builtin, which breaks it).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <run-output-root> [extra sbatch options...]" >&2
    exit 2
fi

OUTPUT_ROOT="$(cd "$1" && pwd)"
shift
EXTRA_SBATCH_ARGS=("$@")

ROCPROF_DIR="$OUTPUT_ROOT/rocprof"
RUN_DIR="$(dirname "$OUTPUT_ROOT")"

if [ ! -d "$ROCPROF_DIR" ]; then
    echo "error: no rocprof directory under $OUTPUT_ROOT" >&2
    exit 1
fi

# --- locate the image (resolved.json first, then container.toml) -----------
IMAGE=""
if [ -f "$RUN_DIR/resolved.json" ]; then
    IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_path"])' \
        "$RUN_DIR/resolved.json" 2>/dev/null || true)"
fi
if [ -z "$IMAGE" ] && [ -f "$RUN_DIR/container.toml" ]; then
    IMAGE="$(sed -n 's/^image *= *"\(.*\)"/\1/p' "$RUN_DIR/container.toml")"
fi
if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
    echo "error: could not resolve a usable image path (got: '$IMAGE')" >&2
    exit 1
fi

# --- collect the databases; the list uses in-container paths ---------------
# The output root is mounted at /apertus/output inside the container, so a
# host path $OUTPUT_ROOT/... maps to /apertus/output/...
DBS=()
while IFS= read -r f; do DBS+=("$f"); done \
    < <(find "$ROCPROF_DIR" -mindepth 2 -maxdepth 2 -name '*_results.db' | sort)

if [ ${#DBS[@]} -eq 0 ]; then
    echo "error: no *_results.db files under $ROCPROF_DIR" >&2
    exit 1
fi
echo "found ${#DBS[@]} database(s) under $ROCPROF_DIR"
echo "image: $IMAGE"

DBS_LIST="$OUTPUT_ROOT/pftrace-export.dbs.txt"
printf '%s\n' "${DBS[@]}" | sed "s|^$OUTPUT_ROOT|/apertus/output|" > "$DBS_LIST"

# --- write the EDF (mirrors the run's container.toml, rw output mount) -----
EDF="$OUTPUT_ROOT/pftrace-export.container.toml"
cat > "$EDF" <<EOF
image = "$IMAGE"
workdir = "/apertus/output"
entrypoint = false
mounts = ["$OUTPUT_ROOT:/apertus/output"]

[annotations]
com.hooks.cxi.enabled = "false"
com.hooks.aws_ofi_nccl.enabled = "false"
com.hooks.netstack.source = "host"
EOF
echo "edf:   $EDF"

# --- in-container conversion loop ------------------------------------------
# rocpd2pftrace -i <db> -o <base> -d <dir> writes <dir>/<base>.pftrace.
# Each database gets its pftrace written next to it (same node subdirectory).
INNER_SCRIPT="$OUTPUT_ROOT/pftrace-export.inner.sh"
cat > "$INNER_SCRIPT" <<'EOF'
#!/bin/bash
set -uo pipefail
while IFS= read -r db; do
    [ -n "$db" ] || continue
    dir="$(dirname "$db")"
    base="$(basename "$db" _results.db)"
    if [ -f "$dir/$base.pftrace" ]; then
        echo "SKIP $db (pftrace exists)"
        continue
    fi
    echo "== $db"
    # Match the capture configuration (rocprofv3 ran with a 2 GiB perfetto buffer).
    if /opt/venv/bin/rocpd2pftrace -i "$db" -o "$base" -d "$dir" \
        --perfetto-buffer-size 2097152; then
        echo "OK   $dir/$base.pftrace"
    else
        echo "FAIL $db (exit $?)" >&2
    fi
done < /apertus/output/pftrace-export.dbs.txt
EOF
chmod +x "$INNER_SCRIPT"

# --- submit ----------------------------------------------------------------
JOB_NAME="pftrace-export-$(basename "$RUN_DIR")"
echo "submitting conversion job..."
sbatch --parsable \
    --job-name="$JOB_NAME" \
    --account=csstaff \
    --partition=mi300 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=64G \
    --time=04:00:00 \
    --reservation=cscs \
    --output="$OUTPUT_ROOT/pftrace-export.log" \
    "${EXTRA_SBATCH_ARGS[@]}" \
    --wrap="srun --environment=$EDF bash /apertus/output/pftrace-export.inner.sh"

echo "job submitted. log: $OUTPUT_ROOT/pftrace-export.log"
