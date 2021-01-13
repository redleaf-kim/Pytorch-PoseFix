from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys, os

def add_path(path):
    if path not in sys.path:
        # print(path)
        sys.path.append(path)

this_dir = os.path.dirname(__file__)

for add in ["nms", "utils", "data"]:
    add_path(os.path.join(this_dir, os.pardir, add))

from .coco_dataset import COCODataset as coco