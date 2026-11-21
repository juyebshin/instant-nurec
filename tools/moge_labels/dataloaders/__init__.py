from .factory import get_dataset

# Importing a loader registers it. Only the three datasets the PMT fine-tune labels are kept (DVGT also has
# kitti, open_scene, ddad, nuplan, navsim loaders).
from . import argoverse_loader
from . import nuscene_loader
from . import waymo_loader
