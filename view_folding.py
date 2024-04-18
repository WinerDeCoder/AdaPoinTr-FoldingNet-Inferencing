import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F




location_folding = glob('folding_images/z_aware/coarse_intense/demo/storage/room_*/partial/medium/*/*/fine.npy')

location = glob('z_aware/coarse_intense/demo/storage/room_*/partial/medium/*/*/fine.npy')

#print(location_folding, location)

location_partial = glob('demo/storage/room_*/partial/medium/*/*.npy')

location_full = glob('demo/storage/room_*/full/*.npy')


location_folding = sorted(location_folding)
location = sorted(location)
location_partial = sorted(location_partial)
location_full = sorted(location_full)

# print(len(location_full), len(location), len(location_partial))
# exit()


for i in range(0, len(location), 15):
    partial_dict = {}
    complete_dict = {}
    partial_part = location_partial[i:i+15]
    location_part = location[i:i+15]
    location_folding_part = location_folding[i:i+15]

    print(location_part[0][41:])
        
    for i in range(len(location_part)):
        objecter = np.load(location_part[i])
        #location_part[i] = location_part[i].split("/")
        #complete_dict[f"{location_part[i][-2]}"] = objecter
        
        objecter_folding = np.load(location_folding_part[i])
        

        object_partial = np.load(partial_part[i])
        #partial_part[i] = partial_part[i].split("/")
        #partial_dict[f"{partial_part[i][-1]}"] = object_partial

        object_full = np.load(location_full[i%16])
        #partial_part[i] = partial_part[i].split("/")
        #partial_dict[f"{partial_part[i][-1]}"] = object_partial


        objecter = torch.tensor(objecter) 
        objecter_folding = torch.tensor(objecter_folding) +torch.tensor([-1.2,0,0]) 
        object_partial = torch.tensor(object_partial) +torch.tensor([2.4,0,0]) 
        object_full = torch.tensor(object_full) +torch.tensor([3.6,0,0]) 

        combine = torch.cat((objecter, objecter_folding, object_partial, object_full))
        color = torch.tensor([1,2,3,4])


        GeometricTools.drawPointCloudsColorsClasses( combine,  color [[0,0,0]])


    # totaler = np.concatenate(tuple(partial_dict.values()), axis = 0)
        
    # colour = [torch.ones(torch.tensor(partial_dict[f"{list(partial_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(partial_dict.keys()))]

    # color = torch.cat(tuple(colour), dim = 0

    # totaler_complete = np.concatenate(tuple(complete_dict.values()), axis = 0) + np.array([1,0,0])* 7
        
    # colour_complete = [torch.ones(torch.tensor(complete_dict[f"{list(complete_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(complete_dict.keys()))]

    # color_complete = torch.cat(tuple(colour_complete), dim = 0)

    # GeometricTools.drawPointCloudsColorsClasses( torch.cat((torch.tensor(totaler), torch.tensor(totaler_complete))),  torch.cat((color, color_complete)), [[0,0,0]])

    #plj-y57NugfA7yh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

    partial_dict = {}
    complete_dict = {}