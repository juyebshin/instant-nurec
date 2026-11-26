#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# OpenScene geo-split index for pmt_openscene_motion.yaml: $INDEX_DIR/geo/openscene/{train,val}/<log>_<a>-<b>.pkl.
# Split: configs/splits/geo_near/openscene (runs <log>:<first>-<last> of the 240 MoGe-3-labelled logs). Runs
# under 8 frames are skipped: no T=8 context window at 2 Hz fits in them (train 53 of 496, val 29 of 336).
# Existing pickles are skipped; REEXPORT=1 rewrites them (e.g. once new Mask2Former masks exist under
# data/m2f/openscene/<log>/seg_<CAM>/<token>.png; raw-image or MoGe-grid masks, told apart by size).
# SWEEPS=1 writes the same runs plus nuPlan's 10 Hz in-between frames and ego poses (export_index.py
# --openscene-sweeps; needs scripts/data/fetch_openscene_sweeps.py's download) to $INDEX_DIR/geo/openscene_sweeps
# for pmt_openscene_motion_sweeps.yaml; the 2 Hz tree is untouched.
source "$(dirname "$0")/common.sh"
OUT=$INDEX_DIR/geo/openscene
FLAGS=
[ "${SWEEPS:-0}" = 1 ] && OUT=$INDEX_DIR/geo/openscene_sweeps && FLAGS=--openscene-sweeps
export FLAGS
for split in train val; do
  tr ' ' '\n' < $GEO/openscene/$split.txt | while read run; do
    [ -n "$run" ] || continue
    span=${run##*:}; a=${span%-*}; b=${span#*-}
    [ $((b - a + 1)) -ge 8 ] || continue
    name=${run/:/_}
    [ -f $OUT/$split/$name.pkl ] && [ "${REEXPORT:-0}" != 1 ] || echo "$split $run $name"
  done
done | xargs -P $JOBS -L 1 sh -c '
  mkdir -p '$OUT'/$0 && tmp=$(mktemp -d '$OUT'/../.tmp_os_XXXX) &&
  $EXPORT --dataset openscene --scenes $1 --out-dir $tmp $FLAGS > $tmp.log 2>&1 &&
  mv $tmp/$2.pkl '$OUT'/$0/$2.pkl || { echo "FAILED $1"; tail -3 $tmp.log; }; rm -rf $tmp $tmp.log'
for split in train val; do echo openscene $split $(ls $OUT/$split | wc -l); done
echo ALL_DONE
