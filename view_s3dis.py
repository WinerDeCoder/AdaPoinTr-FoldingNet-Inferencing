import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F


location = glob('s3dis/*/fine.npy')


location_full = glob('demo/s3dis_npy/*.npy')

location = sorted(location)
location_full = sorted(location_full)


for i in range(len(location)):
    
    partial = np.load(location[i])
    full = np.load(location_full[i])
    
    full[:,0] = full[:,0] - np.mean(full[:,0]) + 2
    full[:,2] = full[:,2] - np.mean(full[:,2]) + 2
    GeometricTools.drawPointCloudsColorsClasses( torch.tensor(full), torch.tensor([3]), [[0,0,0]])
    
    #GeometricTools.drawPointCloudsColorsClasses( torch.cat((torch.tensor(partial), torch.tensor(full) + torch.tensor([5.2,0,0]))), torch.tensor([1,2]), [[0,0,0]])
    
    
    
    