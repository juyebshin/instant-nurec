#!/usr/bin/env bash
# Copied from DVGT output/moge3_relabel/run.sh (its nuScenes entries) (commit 51cf3f6, working tree as run 2026-11); only paths changed.
# MoGe-3 relabel of the DVGT replay datasets on one GPU, one dataset split at a time.
# Rerunning skips finished splits (main.py itself cannot resume inside a split).
# Usage: GPU=3 bash run.sh [dataset:split ...]     (default queue below)
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OUT=data/moge3/runs/nuscenes
ROOT=data/moge3      # -> $ROOT/{image,proj_depth,pred_depth,align_depth}/<dataset>/...
META=data/moge3/meta            # -> <dataset>_<split>.parquet
GPU=${GPU:-3}
V3_PY=/home1/irteam/workspace/venv-dvgt-moge3/bin/python
QUEUE=("$@")
[ ${#QUEUE[@]} -gt 0 ] || QUEUE=(nuscene:train nuscene:val)
log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p $OUT

export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for item in "${QUEUE[@]}"; do
  ds=${item%%:*}; split=${item#*:}
  if [ -f $OUT/${ds}_${split}.done ]; then log "$ds $split already done"; continue; fi
  log "label $ds $split on GPU $GPU"
  t0=$(date +%s)
  $V3_PY -m torch.distributed.run --nproc_per_node=1 --master_port=29642 tools/moge_labels/main.py $ds \
    --split=$split --batch_size=2 --num_workers=8 --save_workers=16 \
    --moge_version=3 --ckpt_path=checkpoints/moge-3-vitl/model.pt \
    --storage_root=$ROOT --meta_storage_path=$META --log_dir=$OUT/label_logs \
    --gen_meta --gen_image --gen_proj_depth --gen_pred_depth --gen_align_depth
  [ -f $META/${ds}_${split}.parquet ] || { log "ERROR: $ds $split wrote no meta parquet"; exit 1; }
  echo "${ds}_${split} $(( $(date +%s) - t0 ))s" >> $OUT/timing.txt
  touch $OUT/${ds}_${split}.done
done
log "done"
touch $OUT/DONE
