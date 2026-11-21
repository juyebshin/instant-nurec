#!/usr/bin/env bash
# Copied from DVGT output/waymo_moge3/run.sh (commit 51cf3f6, working tree as run 2026-11); only paths changed.
# NOTE: data/waymo_ns -> /home1/irteam/data/waymo_ns, the only copy (DVGT links to it too); preprocessing writes there.
# Waymo for the DVGT-2 fine-tune: preprocess official training (798) + validation (202) segments into
# data/waymo_ns (CPU, TF env), then MoGe-3-label them at 2 Hz (loader splits train_2hz / val_2hz) on one GPU.
# Runs alongside the OpenScene relabel on GPU 3. Rerunnable: preprocessing skips extracted segments, labeling skips
# finished splits (inside a split main.py cannot resume). Usage: GPU=2 bash run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OUT=data/moge3/runs/waymo_moge3
NS=data/waymo_ns
ROOT=data/moge3      # -> $ROOT/{image,proj_depth,pred_depth,align_depth}/waymo/<segment>/...
META=data/moge3/meta            # -> waymo_<split>.parquet
GPU=${GPU:-2}
WAYMO_PY=/home1/irteam/miniconda3/envs/sjy-waymo/bin/python
V3_PY=/home1/irteam/workspace/venv-dvgt-moge3/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p $OUT

# 1. preprocess (no testing split: the loader treats every non-validation segment as train)
for s in training validation; do
  if [ -f $OUT/preprocess_$s.done ]; then log "preprocess $s already done"; continue; fi
  log "preprocess $s"
  CUDA_VISIBLE_DEVICES= $WAYMO_PY tools/moge_labels/preprocess/waymo.py -i /home1/irteam/data/waymo -o $NS \
    --split $s --num_workers 16 2>&1 | grep -vE "it/s\]|s/it\]|^\s*$|tensorflow" || true
  touch $OUT/preprocess_$s.done
done
n=$(find $NS -mindepth 2 -maxdepth 2 -name transforms.json | wc -l)
[ "$n" -eq 1000 ] || { log "ERROR: $n segments with transforms.json in $NS, expected 1000 (rerun to retry failed ones)"; exit 1; }

# 2. label at 2 Hz
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for split in train_2hz val_2hz; do
  if [ -f $OUT/waymo_$split.done ]; then log "waymo $split already done"; continue; fi
  log "label waymo $split on GPU $GPU"
  t0=$(date +%s)
  $V3_PY -m torch.distributed.run --nproc_per_node=1 --master_port=29643 tools/moge_labels/main.py waymo \
    --split=$split --batch_size=2 --num_workers=8 --save_workers=16 \
    --moge_version=3 --ckpt_path=checkpoints/moge-3-vitl/model.pt \
    --storage_root=$ROOT --meta_storage_path=$META --log_dir=$OUT/label_logs \
    --gen_meta --gen_image --gen_proj_depth --gen_pred_depth --gen_align_depth
  [ -f $META/waymo_$split.parquet ] || { log "ERROR: waymo $split wrote no meta parquet"; exit 1; }
  echo "waymo_$split $(( $(date +%s) - t0 ))s" >> $OUT/timing.txt
  touch $OUT/waymo_$split.done
done
log "done"
touch $OUT/DONE
