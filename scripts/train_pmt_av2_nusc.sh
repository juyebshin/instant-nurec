#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# AV2 + nuScenes fine-tune (docs/PMT_FINETUNE.md §3): configs/training/pmt_av2_nusc.yaml,
# 4 GPUs x batch 2 (total 8), lr 5.7e-5, 2 epochs (2454 steps). Peak ~80 GB per GPU on AV2 batches.
#
#   GPUS=0,1,2,3 scripts/train_pmt_av2_nusc.sh                                 # foreground
#   setsid nohup scripts/train_pmt_av2_nusc.sh > /dev/null 2>&1 &         # detached (shared machine)
#
# Self-contained: needs only the sjy-inurec env (with its conda nvcc 12.8 + gcc 11.4) and this repo.
# Checkpoints go to checkpoints/training/<RUN_ID>/ (config out_dir, relative to the repo root), the
# console log to checkpoints/training/<RUN_ID>.log, and the gsplat JIT build to .cache/ (first run
# compiles it, a few minutes). All three are gitignored.
#
# GPUS must list as many GPUs as the config's system.devices; check nvidia-smi first, since a GPU
# shared with another job slows every rank. To resume after a crash, copy the config, add
# resume_from_checkpoint: checkpoints/training/<RUN_ID>/checkpoints/last.ckpt, and keep the GPU count.
set -euo pipefail

GPUS=${GPUS:-0,1,2,3}
CONFIG=${CONFIG:-configs/training/pmt_av2_nusc.yaml}
RUN_ID=${RUN_ID:-pmt-av2-nusc-2ep}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOG=${LOG:-checkpoints/training/$RUN_ID.log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

set +u  # conda's (de)activate hooks read unset variables
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sjy-inurec
set -u
# gsplat JIT: the env's own nvcc / gcc (conda activate sets CC/CXX); build cache inside the repo.
# TORCH_CUDA_ARCH_LIST unset = build for the visible GPUs (H200 9.0, B-series 10.0, Pro 6000 12.0).
export CUDA_HOME=$CONDA_PREFIX TORCH_EXTENSIONS_DIR=$REPO/.cache/torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Without these, multi-GPU DDP hangs at a random step inside NCCL's non-blocking enqueue.
export TORCH_NCCL_USE_COMM_NONBLOCKING=0 NCCL_COMM_BLOCKING=1 PYTHONFAULTHANDLER=1

echo "$(date -Is) run=$RUN_ID gpus=$GPUS config=$CONFIG"
INSTANT_NUREC_RUN_ID=$RUN_ID CUDA_VISIBLE_DEVICES=$GPUS instant-nurec-train --config "$CONFIG"
