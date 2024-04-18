import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F


location = glob('s3dis/*/fine.npy')

location_folding = glob('s3dis_folding/*/fine.npy')


location_full = glob('demo/s3dis_npy/*.npy')

location_folding = sorted(location_folding)
location = sorted(location)
location_full = sorted(location_full)


for i in range(len(location)):
    
    folding = np.load(location_folding[i])
    partial = np.load(location[i])
    full = np.load(location_full[i])
    
    GeometricTools.drawPointCloudsColorsClasses( torch.cat((torch.tensor(folding) - torch.tensor([2.2,0,0]), torch.tensor(partial), torch.tensor(full) + torch.tensor([2.2,0,0]))), torch.tensor([1,2]), [[0,0,0]])
    
    
    
    