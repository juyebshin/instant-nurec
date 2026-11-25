#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Geo-split ("near") tree for the AV2 + nuScenes + Waymo fine-tune: $INDEX_DIR/geo/<dataset>/{train,val}/.
# Splits: configs/splits/geo_near/{av2,nusc,waymo} (README there). Geo-test is not indexed (held out).
#   AV2 geo-train (695): symlinks to the official-split pickles in $INDEX_DIR/av2/ (run export_av2.sh first).
#   AV2 geo-val   (150): exported here.
#   nuScenes: symlinks to $INDEX_DIR/nusc/ (export_nusc.sh first); official-test scenes have no MoGe depth and
#   are skipped (geo train 554 of 657, val 99 of 124).
#   Waymo geo train / val (700 / 151): exported here.
# Existing pickles are skipped; REEXPORT_WAYMO=1 rewrites the Waymo ones (e.g. once new masks exist).
source "$(dirname "$0")/common.sh"
OUT=$INDEX_DIR/geo

link() {  # link <dataset> <split> <list>: symlink every listed pickle found under $INDEX_DIR/<dataset>/*/
  mkdir -p $OUT/$1/$2
  while read name; do
    src=$(ls $INDEX_DIR/$1/*/$name.pkl 2>/dev/null | head -1)
    [ -n "$src" ] && ln -sfn ../../../${src#$INDEX_DIR/} $OUT/$1/$2/$name.pkl
  done < $3
}
link av2 train $GEO/av2/train.txt
link nusc train $GEO/nusc/train.txt
link nusc val $GEO/nusc/val.txt

{ sed 's/^/av2 val /' $GEO/av2/val.txt
  sed 's/^/waymo train /' $GEO/waymo/train.txt
  sed 's/^/waymo val /' $GEO/waymo/val.txt
} | while read ds split name; do
    [ -f $OUT/$ds/$split/$name.pkl ] && ! [ "$ds" = waymo -a "${REEXPORT_WAYMO:-0}" = 1 ] || echo "$ds $split $name"
  done \
  | xargs -P $JOBS -L 1 sh -c '
  mkdir -p '$OUT'/$0/$1 && tmp=$(mktemp -d '$OUT'/.tmp_XXXX) &&
  $EXPORT --dataset $0 --split auto --scenes $2 --out-dir $tmp &&
  mv $tmp/$2.pkl '$OUT'/$0/$1/$2.pkl || echo "FAILED $0 $1 $2"; rm -rf $tmp'
for ds in av2 nusc waymo; do for split in train val; do echo $ds $split $(ls $OUT/$ds/$split | wc -l); done; done
echo ALL_DONE
