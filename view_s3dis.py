import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F


location = glob('denoise/radius/s3dis_attn/*/fine.npy')


location_full = glob('demo/s3dis_npy/*.npy')

#location = sorted(location)
#location_full = sorted(location_full)


for i in range(len(location)):
    
    partial = np.load(location[i])
    full = np.load(location_full[i])
    
    if location_full[i][15] != "t":
        continue
    
    full[:,0] = full[:,0] - np.mean(full[:,0])  + 1
    full[:,2] = full[:,2] - np.mean(full[:,2]) 
    full[:,1] = full[:,1] +1.3
    
    partial[:,0] = partial[:,0] - np.mean(partial[:,0])  + 1
    partial[:,2] = partial[:,2] - np.mean(partial[:,2]) 
    partial[:,1] = partial[:,1] +1.3
    #GeometricTools.drawPointCloudsColorsClasses( torch.tensor(full), torch.tensor([3]), [[0,0,0]])
    
    GeometricTools.drawPointCloudsColorsClasses( torch.cat((torch.tensor(partial), torch.tensor(full) + torch.tensor([3,0,0]))), torch.tensor([1,2]), [[0,0,0]])
    
    
    
    