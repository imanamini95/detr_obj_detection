import os
import sys

sys.path.insert(0, os.getcwd())
from scripts.models.detr import build


def build_model(args):
    return build(args)
