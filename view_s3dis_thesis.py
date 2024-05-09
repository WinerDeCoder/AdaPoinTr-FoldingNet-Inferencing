import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F


location = glob('result_s3room/pointattn/3D-Wall-Completer/all_room/*/*/fine.npy')


location_full = glob('full_room_s3dis/*/*.npy')

location = sorted(location)
location_full = sorted(location_full)


for i in range(len(location)):
    
    partial = np.load(location[i])
    full = np.load(location_full[i])
    
    #GeometricTools.drawPointCloudsColorsClasses( torch.cat((torch.tensor(partial), torch.tensor(full) + torch.tensor([2.2,0,0]))), torch.tensor([1,2]), [[0,0,0]])
    print(location_full[i])
    GeometricTools.drawPointCloudsColorsClasses( torch.tensor(full) , torch.tensor([1,2]), [[0,0,0]])
    
    