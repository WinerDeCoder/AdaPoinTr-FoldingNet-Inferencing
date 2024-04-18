import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F


location = glob('inference_z_aware/Info/demo/storage/room_*/partial/medium/*/*/fine.npy')

location_partial = glob('demo/storage/room_*/partial/medium/*/*.npy')
#print(location)
location_full = glob('demo/storage/room_*/full/*.npy')

location = sorted(location)
location_partial = sorted(location_partial)
location_full = sorted(location_full)


for i in range(0, len(location), 15):
    partial_dict = {}
    complete_dict = {}
    full_dict = {}
    partial_part = location_partial[i:i+15]
    location_part = location[i:i+15]
    full_part = location_part.copy()
    print(location_part[0][41:])
        
    for i in range(len(location_part)):

        objecter = np.load(location_part[i])
        location_part[i] = location_part[i].split("/")
        complete_dict[f"{location_part[i][-2]}"] = objecter

        object_partial = np.load(partial_part[i])
        partial_part[i] = partial_part[i].split("/")
        partial_dict[f"{partial_part[i][-1]}"] = object_partial

        object_full = np.load(location_full[i%15])
        full_part[i%15] = location_full[i%15].split("/")
        full_dict[f"{full_part[i%16][-1]}"] = object_full
        
        


    totaler = np.concatenate(tuple(partial_dict.values()), axis = 0) + np.array([1,0,0])* 7
        
    colour = [torch.ones(torch.tensor(partial_dict[f"{list(partial_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(partial_dict.keys()))]

    color = torch.cat(tuple(colour), dim = 0)


    totaler_complete = np.concatenate(tuple(complete_dict.values()), axis = 0) 
        
    colour_complete = [torch.ones(torch.tensor(complete_dict[f"{list(complete_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(complete_dict.keys()))]

    color_complete = torch.cat(tuple(colour_complete), dim = 0)


    totaler_full = np.concatenate(tuple(full_dict.values()), axis = 0) + np.array([1,0,0])* 14
        
    colour_full = [torch.ones(torch.tensor(full_dict[f"{list(full_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(full_dict.keys()))]

    color_full = torch.cat(tuple(colour_full), dim = 0)

    GeometricTools.drawPointCloudsColorsClasses( torch.cat((torch.tensor(totaler), torch.tensor(totaler_complete), torch.tensor(totaler_full))),  torch.cat((color, color_complete, color_full)), [[0,0,0]])



    partial_dict = {}
    complete_dict = {}