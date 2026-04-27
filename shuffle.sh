#!/bin/bash -eu
set -o pipefail
{
# Shuffles all NPZ data for nano training, splitting into train/val sets.
# Usage: bash shuffle.sh INPUTDIR OUTPUTDIR TMPDIR NTHREADS [extra shuffle.py args...]
#
# Examples:
#   # Default: val = 5% of data (MD5 hash split)
#   bash shuffle.sh /data/npz /out /tmp 8
#
#   # Fixed val size: val = 2 files (2 * 131072 rows)
#   VAL_NUM_FILES=2 bash shuffle.sh /data/npz /out /tmp 8
#
#   # Limit memory-heavy sharding, keep merge moderately parallel
#   bash shuffle.sh /data/npz /out /tmp 24 --shard-processes 4 --merge-processes 8
#
#   # Reuse scan results across reruns
#   bash shuffle.sh /data/npz /out /tmp 12 --scan-cache /tmp/katago_scan_cache.sqlite
#
#   # Cap local scratch usage by merging every 32 worker groups
#   bash shuffle.sh /data/npz /out /tmp 12 --max-active-worker-groups 32
#
#   # Persist wave progress and resume after crashes/restarts
#   bash shuffle.sh /data/npz /out /tmp 12 --max-active-worker-groups 32 --shard-cache
#
#   # Compress intermediate shards (saves tmp disk, slower)
#   COMPRESS_SHARDS=1 bash shuffle.sh /data/npz /out /tmp 8

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 INPUTDIR OUTPUTDIR TMPDIR NTHREADS [extra args...]"
    echo "INPUTDIR   directory containing NPZ files (searched recursively)"
    echo "OUTPUTDIR  output directory, will contain train/ and val/ subdirectories"
    echo "TMPDIR     scratch space, ideally on fast local disk"
    echo "NTHREADS   number of parallel processes for shuffling"
    exit 0
fi
INPUTDIR="$1"
shift
OUTPUTDIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"

VAL_NUM_FILES_ARG=""
if [[ -n "${VAL_NUM_FILES:-}" ]]; then
    VAL_NUM_FILES_ARG="--val-num-files $VAL_NUM_FILES"
fi

COMPRESS_SHARDS_ARG=""
if [[ -n "${COMPRESS_SHARDS:-}" ]]; then
    COMPRESS_SHARDS_ARG="--compress-shards"
fi

#------------------------------------------------------------------------------

mkdir -p "$OUTPUTDIR"
mkdir -p "$TMPDIR"/train
mkdir -p "$TMPDIR"/val

echo "Beginning shuffle at $(date "+%Y-%m-%d %H:%M:%S")"

time python3 "$SCRIPTDIR"/shuffle.py \
     "$INPUTDIR" \
     --num-processes "$NTHREADS" \
     --rows-per-file 131072 \
     --split "train:0.00:0.95:$OUTPUTDIR/train:$TMPDIR/train" \
     --split "val:0.95:1.00:$OUTPUTDIR/val:$TMPDIR/val" \
     $VAL_NUM_FILES_ARG \
     $COMPRESS_SHARDS_ARG \
     "$@" \
     2>&1 | tee "$OUTPUTDIR"/output.txt

echo "Finished shuffle at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Output: $OUTPUTDIR/{train,val}/"
echo ""

exit 0
}
