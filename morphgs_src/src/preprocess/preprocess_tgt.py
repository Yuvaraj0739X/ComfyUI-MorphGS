# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import os
import sys
import torch

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
geo_aware_path = os.path.join(base_path, "extlibs", "GeoAware")
if geo_aware_path not in sys.path:
    sys.path.append(geo_aware_path)

from utils.render_utils import render_mesh_with_pytorch3d
from utils.feature_utils import extract_feature, set_feature_extraction


IMSIZE = 256
NUM_PATCHES = 60
# load model
with torch.no_grad():
    sd_model, sd_aug, extractor_vit, aggre_net, num_patches = set_feature_extraction(NUM_PATCHES)

def render_char_360(tgt_dir):
    mesh_path = os.path.join(tgt_dir, "mesh.obj") 
    texture_path = os.path.join(tgt_dir, "texture.png")
    
    if not os.path.exists(texture_path):
        print(f"❌ Texture not found: {texture_path}. Skipping...")
        texture_path = None
    
    if not os.path.exists(os.path.join(tgt_dir, "color")) or not os.listdir(os.path.join(tgt_dir, "color")):
        color_dir, _, normal_dir, _, _, _, _ = render_mesh_with_pytorch3d(
            mesh_path, tgt_dir, (IMSIZE, IMSIZE), use_texture=(texture_path is not None),
            texture_path=texture_path, render_normal=True)
    else:
        print(f"✅ Rendered images already exist at: {os.path.join(tgt_dir, 'color')}. Skipping rendering.")
        color_dir = os.path.join(tgt_dir, "color")
        normal_dir = os.path.join(tgt_dir, "normal")
        
    return color_dir, normal_dir

def extract_char_features(tgt_dir, normal_dir):
    out_dir = os.path.join(tgt_dir, "feature")

    if not os.path.exists(out_dir) or not os.listdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        extract_feature(normal_dir, out_dir, sd_model, sd_aug, aggre_net, extractor_vit, num_patches)
    else:
        print(f"✅ Features already exist at: {out_dir}. Skipping extraction.") 
        
    return out_dir


if __name__ == "__main__":
    """
     preprocess target object for motion transfer:

     1. render target object from multiple views
     2. extract 2d features and save.
    """

    # first argument will be the path to the target object
    tgt_dir = sys.argv[1]

    os.makedirs(tgt_dir, exist_ok=True)
    tgt_char_name = os.path.basename(tgt_dir)

    # Render 360 views
    color_dir, normal_dir = render_char_360(tgt_dir)

    # Extract Features
    feature_dir = extract_char_features(tgt_dir, normal_dir)
