# SPDX-License-Identifier: Apache-2.0
"""Mapillary Vistas v2 id -> KelvinSemanticClass, the ``semantic_lut`` stored in every index pickle.

The masks are Mask2Former Vistas v2 predictions (124 ids, names from Mask2Former
``datasets/mapillary_vistas_config_v2.0.json``). This table is the exact output of the PMT exporter's
``semantic_lut()`` as of 2026-11-25, written out as data: it equals the ``semantic_lut`` in the existing
AV2 / nuScenes / Waymo / OpenScene pickles. Its LANE / CROSSWALK / ROAD / BOUNDARY assignment follows the
road-map remap of RoGS's ``MapillaryVistasV2`` (PseudoMapTrainer ``RoGS/datasets/label_mappings/
mapillary_vistas.py``, AGPL-3.0), so both models agree on the map classes; no code is copied from it.
Ids not listed are OTHERS.
"""

import numpy as np

# KelvinSemanticClass order (instant_nurec/primitives/kelvin_primitive.py); 255 = ignore
OTHERS, EGO, SKY, ROAD, MOVABLE, LANE, CROSSWALK, BOUNDARY, IGNORE = 0, 1, 2, 3, 4, 5, 6, 7, 255

VISTAS_V2_TO_CLASS = {
    EGO: {118: "void--car-mount", 120: "void--ego-vehicle"},
    SKY: {61: "nature--sky"},
    ROAD: {
        13: "construction--flat--bike-lane", 21: "construction--flat--road", 22: "construction--flat--road-shoulder",
        23: "construction--flat--service-lane", 39: "marking--discrete--arrow--left",
        40: "marking--discrete--arrow--other", 41: "marking--discrete--arrow--right",
        42: "marking--discrete--arrow--split-left-or-straight", 43: "marking--discrete--arrow--split-right-or-straight",
        44: "marking--discrete--arrow--straight", 46: "marking--discrete--give-way-row",
        47: "marking--discrete--give-way-single", 50: "marking--discrete--other-marking",
        51: "marking--discrete--stop-line", 52: "marking--discrete--symbol--bicycle",
        53: "marking--discrete--symbol--other", 54: "marking--discrete--text",
        57: "marking-only--discrete--other-marking", 58: "marking-only--discrete--text", 69: "object--catch-basin",
        74: "object--manhole",
    },
    MOVABLE: {
        30: "human--person--individual", 31: "human--person--person-group", 32: "human--rider--bicyclist",
        33: "human--rider--motorcyclist", 34: "human--rider--other-rider", 105: "object--vehicle--bicycle",
        106: "object--vehicle--boat", 107: "object--vehicle--bus", 108: "object--vehicle--car",
        109: "object--vehicle--caravan", 110: "object--vehicle--motorcycle", 111: "object--vehicle--on-rails",
        112: "object--vehicle--other-vehicle", 113: "object--vehicle--trailer", 114: "object--vehicle--truck",
        115: "object--vehicle--vehicle-group", 116: "object--vehicle--wheeled-slow",
    },
    LANE: {
        10: "construction--barrier--separator", 35: "marking--continuous--dashed", 36: "marking--continuous--solid",
        37: "marking--continuous--zigzag", 38: "marking--discrete--ambiguous", 55: "marking-only--continuous--dashed",
    },
    CROSSWALK: {
        14: "construction--flat--crosswalk-plain", 45: "marking--discrete--crosswalk-zebra",
        56: "marking-only--discrete--crosswalk-zebra",
    },
    BOUNDARY: {
        4: "construction--barrier--curb", 8: "construction--barrier--road-median",
        9: "construction--barrier--road-side", 15: "construction--flat--curb-cut", 16: "construction--flat--driveway",
        24: "construction--flat--sidewalk", 25: "construction--flat--traffic-island",
        48: "marking--discrete--hatched--chevron", 49: "marking--discrete--hatched--diagonal", 63: "nature--terrain",
    },
    IGNORE: {123: "void--unlabeled"},
}


def semantic_lut() -> np.ndarray:
    """(256,) uint8: Vistas v2 id -> class."""
    lut = np.full(256, OTHERS, np.uint8)
    for cls, ids in VISTAS_V2_TO_CLASS.items():
        lut[list(ids)] = cls
    return lut
