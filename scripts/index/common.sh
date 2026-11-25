# SPDX-License-Identifier: Apache-2.0
# Shared setup for scripts/index/export_*.sh (sourced). Index pickles for PMTIndexDataset, written by
# tools/index_export/export_index.py in sjy-inurec. Moved from PseudoMapTrainer
# RoGS/output/inurec_finetune/logs/export_*.sh on 2026-11-25 (index tree moved to data/index the same day).
#   INDEX_DIR  output tree (default data/index; relative to the repo root)
#   JOBS       parallel exporter processes (default 8)
# A pickle is written to a temp dir and moved into place, so an interrupted run never leaves a truncated one.
# A running training job is unaffected (it loads every pickle at start); a resume would see new ones.
set -o pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set +u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sjy-inurec
INDEX_DIR=${INDEX_DIR:-data/index}
JOBS=${JOBS:-8}
GEO=configs/splits/geo_near
EXPORT="python tools/index_export/export_index.py"
export INDEX_DIR EXPORT
