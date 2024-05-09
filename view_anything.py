import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F


location = glob('bad_cd/*/*/*/*/*/fine.npy')


location_full = glob('fullrooms3/*/*.npy')

location = sorted(location)
location_full = sorted(location_full)


for i in range(len(location)):
    
    partial = np.load(location[i])
    #full = np.load(location_full[i])
    
    GeometricTools.drawPointCloudsColorsClasses( torch.tensor(partial), torch.tensor([1,2]), [[0,0,0]])
    #print(location_full[i])
    #GeometricTools.drawPointCloudsColorsClasses( torch.tensor(full) , torch.tensor([1,2]), [[0,0,0]])
    
    