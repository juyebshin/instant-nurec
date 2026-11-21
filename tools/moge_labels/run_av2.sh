#!/usr/bin/env bash
# Copied from DVGT output/av2_moge3_{geo_near_train,rest}/run.sh (commit 51cf3f6, working tree as run 2026-11); only paths changed.
# AV2 MoGe-3 pseudo-labels for the logs in LOG_LIST, written as split SPLIT. One GPU. DVGT ran it twice:
# SPLIT=geo_near_train (the 695 geo train logs) and SPLIT=geo_rest (the other 305: geo val 150 + test 150 + 5 unlisted).
# Usage: LOG_LIST=<file, one log id per line> SPLIT=<name> GPU=3 bash run_av2.sh
# Preprocesses all logs into one split, then labels it in chunks (symlinked split views) so that a crash only
# redoes one chunk. Rerunning this script skips finished steps. Usage: see above
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SPLIT=${SPLIT:?set SPLIT, e.g. geo_near_train}
OUT=data/moge3/runs/av2_$SPLIT
PROC=data/argoverse2_process
ROOT=data/moge3      # -> $ROOT/{image,proj_depth,pred_depth,align_depth}/argoverse/<log>/<cam>/<ts>
META=data/moge3/meta            # -> argoverse_<split>.parquet
GPU=${GPU:-2}
CHUNK=100
PRE_PY=/home1/irteam/miniconda3/envs/sjy-rogs/bin/python           # has the av2 api
V3_PY=/home1/irteam/workspace/venv-dvgt-moge3/bin/python           # sjy-dvgt-data + upstream MoGe-3
BASE_PY=/home1/irteam/miniconda3/envs/sjy-dvgt-data/bin/python
log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p $OUT
mapfile -t LOGS < ${LOG_LIST:?set LOG_LIST, one AV2 log id per line}

# 1. preprocess (CPU, 16 processes)
if [ ! -f $OUT/preprocess.done ]; then
  log "preprocess ${#LOGS[@]} logs"
  $PRE_PY tools/moge_labels/preprocess/argoverse.py -i /home1/irteam/data/av2 -o $PROC \
    --log_list $OUT/logs.txt --output_split $SPLIT
  touch $OUT/preprocess.done
fi
n=$(find $PROC/$SPLIT -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n" -eq "${#LOGS[@]}" ] || { log "ERROR: $PROC/$SPLIT has $n log dirs, expected ${#LOGS[@]}"; exit 1; }

# 2. chunk views: $PROC/${SPLIT}_partNN/<log> -> ../$SPLIT/<log>
nparts=$(( (${#LOGS[@]} + CHUNK - 1) / CHUNK ))
for ((p = 0; p < nparts; p++)); do
  d=$PROC/$(printf '%s_part%02d' $SPLIT $p)
  if [ ! -d $d ]; then
    mkdir -p $d
    for id in "${LOGS[@]:$((p * CHUNK)):CHUNK}"; do ln -s ../$SPLIT/$id $d/$id; done
  fi
done

# 3. label each chunk (optionally after the job holding the GPU, pid WAIT_PID, exits)
if [ -n "${WAIT_PID:-}" ]; then log "waiting for pid $WAIT_PID"; while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done; fi
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for ((p = 0; p < nparts; p++)); do
  part=$(printf '%s_part%02d' $SPLIT $p)
  if [ -f $OUT/$part.done ]; then log "$part already done"; continue; fi
  log "label $part on GPU $GPU"
  t0=$(date +%s)
  $V3_PY -m torch.distributed.run --nproc_per_node=1 --master_port=29644 tools/moge_labels/main.py argoverse \
    --split=$part --batch_size=2 --num_workers=8 --save_workers=16 \
    --moge_version=3 --ckpt_path=checkpoints/moge-3-vitl/model.pt \
    --storage_root=$ROOT --meta_storage_path=$META --log_dir=$OUT/label_logs \
    --gen_meta --gen_image --gen_proj_depth --gen_pred_depth --gen_align_depth
  [ -f $META/argoverse_$part.parquet ] || { log "ERROR: $part wrote no meta parquet"; exit 1; }
  echo "$part $(( $(date +%s) - t0 ))s" >> $OUT/timing.txt
  touch $OUT/$part.done
done

# 4. one meta parquet for the whole split
$BASE_PY - <<EOF
import glob
import pandas as pd
fs = sorted(glob.glob('$META/argoverse_${SPLIT}_part*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
assert df.filename.is_unique, 'duplicate filenames across parts'
assert df.scene_id.nunique() == ${#LOGS[@]}, f'{df.scene_id.nunique()} logs in meta'
df.to_parquet('$META/argoverse_$SPLIT.parquet', engine='pyarrow', compression='zstd', index=False)
print(f'{len(fs)} parts -> {len(df)} images, {df.scene_id.nunique()} logs')
EOF
log "done"
touch $OUT/DONE
