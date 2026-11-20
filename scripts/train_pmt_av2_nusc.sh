#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# AV2 + nuScenes fine-tune (docs/PMT_FINETUNE.md §3): configs/training/pmt_av2_nusc.yaml,
# 4 GPUs x batch 2 (total 8), lr 5.7e-5, 2 epochs (2454 steps). Peak ~80 GB per GPU on AV2 batches.
#
#   GPUS=0,1,2,3 scripts/train_pmt_av2_nusc.sh                                 # foreground
#   setsid nohup scripts/train_pmt_av2_nusc.sh > <log> 2>&1 &              # detached (shared machine)
#
# GPUS must list as many GPUs as the config's system.devices; check nvidia-smi first, since a GPU
# shared with another job slows every rank. To resume after a crash, copy the config, add
# resume_from_checkpoint: <out_dir>/<run>/checkpoints/last.ckpt, and keep the GPU count.
set -euo pipefail

GPUS=${GPUS:-0,1,2,3}
CONFIG=${CONFIG:-configs/training/pmt_av2_nusc.yaml}
RUN_ID=${RUN_ID:-pmt-av2-nusc-2ep}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sjy-inurec
# gsplat JIT toolchain from sjy-mapnn. Append, not prepend: its Python would shadow this env's.
B=/home/irteam/miniconda3/envs/sjy-mapnn/bin
export CUDA_HOME=/home/irteam/miniconda3/envs/sjy-mapnn PATH=$PATH:$B TORCH_CUDA_ARCH_LIST=9.0 \
       CC=$B/x86_64-conda-linux-gnu-gcc CXX=$B/x86_64-conda-linux-gnu-g++
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Without these, multi-GPU DDP hangs at a random step inside NCCL's non-blocking enqueue.
export TORCH_NCCL_USE_COMM_NONBLOCKING=0 NCCL_COMM_BLOCKING=1 PYTHONFAULTHANDLER=1

cd "$(dirname "$0")/.."
INSTANT_NUREC_RUN_ID=$RUN_ID CUDA_VISIBLE_DEVICES=$GPUS instant-nurec-train --config "$CONFIG"
