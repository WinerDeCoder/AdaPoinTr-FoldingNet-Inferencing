import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F
import os


num_loop =  glob('demo/storage/room_*/partial/*/*')

for i in range(len(num_loop)):  # Adjust the range as needed
    # Replace "your_command" with the actual command
    command = f"python tools/inference.py cfgs/ShapeNet55_models/AdaPoinTr.yaml \
        experiments_ada_general/final/AdaPoinTr/ShapeNet55_models/example/ckpt-best.pth --pc_root {num_loop[i]} \
                --save_vis_img  --out_pc_root AdaPoinTr-FoldingNet-Inferencing/ada_general/final/{num_loop[i]}"
    print(command)
    os.system(command)
    
    
    
# python tools/inference.py cfgs/ShapeNet55_models/AdaPoinTr.yaml \experiments_ada_general/z_aware/Info/AdaPoinTr/ShapeNet55_models/example/ckpt-best.pth --pc_root demo/s3dis_npy \ --save_vis_img  --out_pc_root inference_result/
