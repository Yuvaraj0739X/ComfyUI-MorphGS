# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import os
import random

import numpy as np
import torch

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image_np = np.array(resized_image_PIL)
    if resized_image_np.dtype == np.uint8:
        scale = 255.0
    elif resized_image_np.dtype == np.uint16:
        scale = 65535.0
    else:
        vmax = float(resized_image_np.max()) if resized_image_np.size > 0 else 1.0
        scale = vmax if vmax > 1.0 else 1.0
    resized_image = torch.from_numpy(resized_image_np.astype(np.float32)) / scale
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    

def set_seed(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable autotuner for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"Set to seed {seed}")
