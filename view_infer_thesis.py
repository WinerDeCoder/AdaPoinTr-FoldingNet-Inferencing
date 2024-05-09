import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F


location = glob('thesis_infer/55/easy/*/fine.npy')

location_partial = glob('inferencing/55/easy/*.npy')

location_full = glob('inferencing/55/full/*.npy')

# location = glob('thesis_infer/PCN/*/fine.npy')

# location_partial = glob('inferencing/PCN_par/partial/*.pcd')

# location_full = glob('inferencing/PCN_par/full/*.pcd')

#location = sorted(location)
#location_partial = sorted(location_partial)
#location_full = sorted(location_full)

# print(len(location_full), len(location), len(location_partial))
# exit()


for i in range(0, len(location), 9):
    partial_dict = {}
    complete_dict = {}
    partial_part = location_partial[i:i+15]
    location_part = location[i:i+15]

    print(location_part[0][41:])
        
    for i in range(len(location_part)):
        objecter = np.load(location_part[i])
        #location_part[i] = location_part[i].split("/")
        #complete_dict[f"{location_part[i][-2]}"] = objecter

        object_partial = np.load(partial_part[i])
        #partial_part[i] = partial_part[i].split("/")
        #partial_dict[f"{partial_part[i][-1]}"] = object_partial

        object_full = np.load(location_full[i%16])
        #partial_part[i] = partial_part[i].split("/")
        #partial_dict[f"{partial_part[i][-1]}"] = object_partial


        objecter = torch.tensor(objecter) 
        object_partial = torch.tensor(object_partial) +torch.tensor([1.2,0,0]) 
        object_full = torch.tensor(object_full) +torch.tensor([2.4,0,0]) 

        combine = torch.cat((objecter, object_partial, object_full))
        color = torch.tensor([1,2,3])

        GeometricTools.drawPointCloudsColorsClasses( combine,  color [[0,0,0]])


    # totaler = np.concatenate(tuple(partial_dict.values()), axis = 0)
        
    # colour = [torch.ones(torch.tensor(partial_dict[f"{list(partial_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(partial_dict.keys()))]

    # color = torch.cat(tuple(colour), dim = 0

    # totaler_complete = np.concatenate(tuple(complete_dict.values()), axis = 0) + np.array([1,0,0])* 7
        
    # colour_complete = [torch.ones(torch.tensor(complete_dict[f"{list(complete_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(complete_dict.keys()))]

    # color_complete = torch.cat(tuple(colour_complete), dim = 0)

    # GeometricTools.drawPointCloudsColorsClasses( torch.cat((torch.tensor(totaler), torch.tensor(totaler_complete))),  torch.cat((color, color_complete)), [[0,0,0]])



    partial_dict = {}
    complete_dict = {}