#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# The AV2 logs in av2_logs.txt (695 = 491 train / 101 val / 103 test, the ones that had MoGe-3 depth on
# 2026-11-21) -> $INDEX_DIR/av2/<split>/<log>.pkl. All 1000 AV2 logs have depth now, but pmt_av2_pilot.yaml and
# pmt_av2_nusc.yaml read these directories whole, so adding logs changes their training data; extend the list
# deliberately. Existing pickles are skipped. The geo-split AV2 train pickles are symlinks to these (export_geo.sh).
source "$(dirname "$0")/common.sh"
grep -v '^#' scripts/index/av2_logs.txt | while read split log; do
  [ -f $INDEX_DIR/av2/$split/$log.pkl ] || echo "$split $log"
done | xargs -P $JOBS -L 1 sh -c '
  mkdir -p $INDEX_DIR/av2/$0 && tmp=$(mktemp -d $INDEX_DIR/av2/.tmp_XXXX) &&
  $EXPORT --dataset av2 --split $0 --scenes $1 --out-dir $tmp &&
  mv $tmp/$1.pkl $INDEX_DIR/av2/$0/$1.pkl || echo "FAILED av2 $0 $1"; rm -rf $tmp'
for split in train val test; do echo av2 $split $(ls $INDEX_DIR/av2/$split | wc -l); done
echo ALL_DONE
