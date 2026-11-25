#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Every nuScenes v1.0-trainval scene (official split: 700 train / 150 val) -> $INDEX_DIR/nusc/<split>/<scene>.pkl.
# 25 scenes per process (loading the devkit tables is the slow part). Existing pickles are skipped. The
# geo-split nuScenes pickles are symlinks to these (export_geo.sh).
source "$(dirname "$0")/common.sh"
for split in train val; do
  mkdir -p $INDEX_DIR/nusc/$split
  python -c "from nuscenes.utils import splits; print('\n'.join(splits.$split))" | while read s; do
    [ -f $INDEX_DIR/nusc/$split/$s.pkl ] || echo $s
  done | xargs -P $JOBS -n 25 sh -c '
    tmp=$(mktemp -d $INDEX_DIR/nusc/.tmp_XXXX) &&
    $EXPORT --dataset nusc --scenes "$@" --out-dir $tmp &&
    mv $tmp/*.pkl $INDEX_DIR/nusc/'$split'/ || echo "FAILED nusc $*"; rm -rf $tmp' _
done
for split in train val; do echo nusc $split $(ls $INDEX_DIR/nusc/$split | wc -l); done
echo ALL_DONE
