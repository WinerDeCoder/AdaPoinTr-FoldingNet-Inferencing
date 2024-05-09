import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import torch.nn.functional as F

def floor():
    range_x         = 6 
    range_y         = 6
    range_z         = 0.02
    
    max_x           = 7
    min_x           = -2
    max_y           = 7
    min_y           = -2
    max_z           = 4
    min_z           = 0
    
    thick           = 0.03
    point_thick     = 1
    fix_point       = 60000

    percent_xy       = range_x / range_y

    num_point_z     = point_thick
    num_point_x     = int(np.sqrt( fix_point/ num_point_z * percent_xy )) 
    num_point_y     = int(fix_point / num_point_x / num_point_z) + 1

    # interval_x1 = np.random.uniform(min_x, max_x, size = (num_point_x, num_point_z, num_point_y) ) 
    # interval_y1 = np.random.uniform(min_y, max_y, size = (num_point_x, num_point_z, num_point_y) ) 
    # interval_z1 = np.random.uniform(min_z, thick, size = (num_point_x, num_point_z, num_point_y) ) 
    # floor1 = np.stack((interval_x1, interval_z1, interval_y1), axis = 3).reshape(-1,3)[:fix_point]

    interval_x1 = np.linspace(min_x, max_x, num_point_x) 
    interval_y1 = np.linspace(min_y, max_y, num_point_y) 
    interval_z1 = np.linspace(min_z, thick, num_point_z) 

    return np.array([[a,b,c] for a in interval_x1 for b in interval_z1 for c in interval_y1])[:fix_point]


location = glob('denoise/radius/PointAttn/*/demo/storage/room_*/partial/easy/*/*/fine.npy')

location_partial = glob('demo/storage/room_*/partial/medium/*/*.npy')
print(location)
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
        print(location_part[i])
        location_part[i] = location_part[i].split("/")
        complete_dict[f"{location_part[i][-1]}"] = objecter

        object_partial = np.load(partial_part[i])
        partial_part[i] = partial_part[i].split("/")
        partial_dict[f"{partial_part[i][-1]}"] = object_partial

        object_full = np.load(location_full[i%15])
        full_part[i%15] = location_full[i%15].split("/")
        full_dict[f"{full_part[i%16][-1]}"] = object_full
        
        
    print(len(complete_dict))

    totaler = np.concatenate(tuple(partial_dict.values()), axis = 0) + np.array([1,-0.07/0.5,0])* 0.5 
        
    colour = [torch.ones(torch.tensor(partial_dict[f"{list(partial_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(partial_dict.keys()))]

    color = torch.cat(tuple(colour), dim = 0)


    totaler_complete = np.concatenate(tuple(complete_dict.values()), axis = 0) + np.array([1,-0.07/0.5,0])* 0.5
    
    mask = totaler_complete[:,1] > 0
    
    totaler_complete = totaler_complete[mask]
        
    colour_complete = [torch.ones(torch.tensor(complete_dict[f"{list(complete_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(complete_dict.keys()))]

    color_complete = torch.cat(tuple(colour_complete), dim = 0)


    totaler_full = np.concatenate(tuple(full_dict.values()), axis = 0) + np.array([1,-0.07/0.5,0])* 0.5#   +np.array([1,0,0])* 14
        
    colour_full = [torch.ones(torch.tensor(full_dict[f"{list(full_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(full_dict.keys()))]

    color_full = torch.cat(tuple(colour_full), dim = 0)
    
    floorer = torch.tensor(floor())
    
    GeometricTools.drawPointCloudsColorsClasses(  torch.concatenate((floorer,torch.tensor(totaler_complete))),  color_full, [[0,0,0]])

    #GeometricTools.drawPointCloudsColorsClasses( torch.cat((floorer, torch.tensor(totaler), torch.tensor(totaler_complete), torch.tensor(totaler_full))),  torch.cat((color, color_complete, color_full)), [[0,0,0]])



    partial_dict = {}
    complete_dict = {}
    
    
