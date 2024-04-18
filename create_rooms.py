import numpy as np
from glob import glob
import GeometricTools
import open3d as o3d
import seaborn as sns
import torch
import random
import os

'''
We will need:
+ A function can take random objects in table, chair, bed, computer, printer,
+ A function can generate rules ( maybe fix first ) -> this wil be in next phase
+ We don't need wall, floor, random
+ Random_rotation, random_on_floor, random_nextto, random_on, IOU 
+ A final function call allto create room


'''

taken_place = {}
object_dict = {}


def compare_IOU(objecter1, objecter2):
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1  = np.min(objecter1[:,0]), np.max(objecter1[:,0]), np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1])
    min_x2, max_x2, min_y2, max_y2, min_z2, max_z2  = np.min(objecter2[:,0]), np.max(objecter2[:,0]), np.min(objecter2[:,2]), np.max(objecter2[:,2]), np.min(objecter2[:,1]), np.max(objecter2[:,1])

    min_x = min(min_x1, min_x2)
    max_x = max(max_x1, max_x2)
    
    min_y = min(min_y1, min_y2)
    max_y = max(max_y1, max_y2)
    
    min_z = min(min_z1, min_z2)
    max_z = max(max_z1, max_z2)
    
    maxer = max((max_x-min_x), (max_y-min_y), (max_z-min_z))
    
    ranger = 3
    ranger_x = int(ranger * (max_x-min_x)/maxer)
    ranger_y = int(ranger * (max_y-min_y)/maxer)
    ranger_z = int(ranger * (max_z-min_z)/maxer)
    
    interval_x0     = np.linspace(min_x, max_x - (max_x-min_x)/ranger_x, ranger_x).reshape(-1,1)
    interval_x1     = np.linspace(min_x + (max_x-min_x)/ranger_x, max_x, ranger_x).reshape(-1,1)
    interval_x      = np.concatenate((interval_x0, interval_x1), axis=1)
    
    interval_y0     = np.linspace(min_y, max_y - (max_y-min_y)/ranger_y, ranger_y).reshape(-1,1)
    interval_y1     = np.linspace(min_y + (max_y-min_y)/ranger_y, max_y, ranger_y).reshape(-1,1)
    interval_y      = np.concatenate((interval_y0, interval_y1), axis=1)
    
    interval_z0     = np.linspace(min_z, max_z - (max_z-min_z)/ranger_z, ranger_z).reshape(-1,1)
    interval_z1     = np.linspace(min_z + (max_z-min_z)/ranger_z, max_z, ranger_z).reshape(-1,1)
    interval_z      = np.concatenate((interval_z0, interval_z1), axis=1)
    
    interval = np.array([[a,b,c,d,e,f] for a,b in interval_x for c,d in interval_y for e,f in interval_z])
    
    a  = 0
    b  = 0
    c  = 0
    
    ob1_x = objecter1[:,0]
    ob1_y = objecter1[:,2]
    ob1_z = objecter1[:,1]
    
    ob2_x = objecter2[:,0]
    ob2_y = objecter2[:,2]
    ob2_z = objecter2[:,1]
    
    for i in range(interval.shape[0]):
        c+=1
        ob1_true =    np.any( (interval[i,0] <= ob1_x) & (ob1_x <= interval[i,1]) &  # Check first column
                        (interval[i,2] <= ob1_y) & (ob1_y <= interval[i,3])  &
                        (interval[i,4] <= ob1_z) & (ob1_z <= interval[i,5]) )

        if ob1_true == False:
            continue     
        b +=1
        
        ob2_true =    np.any( (interval[i,0] <= ob2_x) & (ob2_x <= interval[i,1]) &  # Check first column
                        (interval[i,2] <= ob2_y) & (ob2_y <= interval[i,3])  &
                        (interval[i,4] <= ob2_z) & (ob2_z <= interval[i,5]) )
        if ob2_true == False:
            continue
        a +=1
    if b != 0 and a / b >= 0.25 :
        return False
    return True



def random_on(objecter1, objecter2, object1_name, object2_name):
    
    objecter = random_rotation(objecter1) 
    taken_place[f"{object1_name}"] = [-999,-999,-999,-999, -999, -999]  # set out of boundary
    
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1  = np.min(objecter1[:,0]), np.max(objecter1[:,0]), np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1])
    min_x2, max_x2, min_y2, max_y2, max_z2          = np.min(objecter2[:,0]), np.max(objecter2[:,0]) ,np.min(objecter2[:,2]), np.max(objecter2[:,2]), np.max(objecter2[:,1])

    size_ob_x       = (max_x1 - min_x1) / 2
    size_ob_y       = (max_y1 - min_y1) / 2
    size_ob_z       = (max_z1 - min_z1) / 2
    
    outsider        = 1
    rand_max_x      = max_x2 - size_ob_x/outsider
    rand_min_x      = min_x2 + size_ob_x/outsider
    
    rand_max_y      = max_y2 - size_ob_y/outsider
    rand_min_y      = min_y2 + size_ob_y/outsider
    
    overlap         = -0.005
    maxer           = max((rand_max_x-rand_min_x), (rand_max_y-rand_min_y))
    
    randnum         = 10
    randnum_x       = int(randnum*(rand_max_x-rand_min_x)/maxer)
    randnum_y       = int(randnum*(rand_max_y-rand_min_y)/maxer)
    random_center_x = np.linspace(rand_min_x, rand_max_x, num=randnum_x)
    
    random_center_y = np.linspace(rand_min_y, rand_max_y, num=randnum_y)
    
    np.random.shuffle(random_center_x)
    np.random.shuffle(random_center_y)
    
    concat_xy       = np.array([[a,b] for a in random_center_x for b in random_center_y ])
    
    z               = max_z2 + size_ob_z + 0.015
    for pair in concat_xy:
        x       = pair[0]
        y       = pair[1]
        count   = 0
        for quarad in taken_place.values():
            if ( x + size_ob_x >= quarad[0] + overlap and  x - size_ob_x <= quarad[1] - overlap)  and (y + size_ob_y >= quarad[2] + overlap  and  y - size_ob_y <= quarad[3] - overlap) and ( z + size_ob_z >= quarad[4] + overlap and  z - size_ob_z <= quarad[5] - overlap):
                break
            
            count +=1
            
        if count == len(taken_place.values()) and count !=0 :
            objecter1[:,0]  = objecter1[:,0] - np.mean(objecter1[:,0]) + x
            objecter1[:,2]  = objecter1[:,2] - np.mean(objecter1[:,2]) + y
            
            objecter1[:,1]  = objecter1[:,1] - np.min(objecter1[:,1]) + np.max(objecter2[:,1])
            
            taken_place[f"{object1_name}"] = [ np.min(objecter1[:,0]), np.max(objecter1[:,0]) ,np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1]) ] 
            
            distances       = np.linalg.norm(np.transpose(np.array([objecter2[:,0], objecter2[:,2]])) - np.array([np.mean(objecter1[:,0]), np.mean(objecter1[:,2])]), axis=1)

            closest_index   = np.argmin(distances)

            closest_points  = objecter2[np.isclose(distances, distances[closest_index], atol=0.05)]
            
            
            max_z_point     = closest_points[np.argmax(closest_points[:, 1])]
            
            objecter1[:,1]  = objecter1[:,1] - np.min(objecter1[:,1]) + max_z_point[1] + 0.005
            
            if compare_IOU(objecter1, objecter2) == False :
                continue
            return objecter1
            
    return objecter



def random_nextto(objecter1, objecter2, object1_name, object2_name, boundary_x, boundary_y):
    taken_place[f"{object1_name}"] = [-999,-999,-999,-999, -999, -999]  # set out of boundary
    #print("name", object1_name)
    
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1, mean_x1, mean_y1  = np.min(objecter1[:,0]), np.max(objecter1[:,0]) ,np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1]), np.mean(objecter1[:,0]), np.mean(objecter1[:,2])

    min_x2, max_x2, min_y2, max_y2    = np.min(objecter2[:,0]), np.max(objecter2[:,0]) ,np.min(objecter2[:,2]), np.max(objecter2[:,2])
    
    size_ob_x   = (max_x1 - min_x1) / 2
    size_ob_y   = (max_y1 - min_y1) / 2
    size_ob_z   = (max_z1 - min_z1) / 2
    
    z           = (min_z1 + max_z1)/2
    
    overlap     = -0.01
    ranger      = 0.3   # next to range
    ranger2     = 0.1
    outsider    = 0.02
    lister      = [1,2,3,4]
    np.random.shuffle(lister)
    
    for position in lister:
        if position ==1: # case left
            #print("POS", position)
            max_x = random.uniform(min_x2 - ranger, min_x2- outsider)
            
            if  max_x - (max_x1- min_x1) <=0:
                #print("no")
                continue
            
            y       = random.uniform(min_y2 + ranger2, max_y2 - ranger2)
            
            center  = ( max_x + max_x - (max_x1- min_x1)) /2 
            shift   = np.array([mean_x1, mean_y1]) - np.array([center, y])
            
            count   = 0

            for quarad in taken_place.values():
                if ( center + size_ob_x >= quarad[0] + overlap and  center - size_ob_x <= quarad[1] - overlap)  and (y + size_ob_y >= quarad[2] + overlap  and  y - size_ob_y <= quarad[3] - overlap)and ( z + size_ob_z >= quarad[4] + overlap and  z - size_ob_z <= quarad[5] - overlap):
                    #print("eror")
                    #print("taken", quarad, taken_place)
                    #print("center xy", center, y)
                    break
                count +=1
            if count == len(taken_place.values()) and count !=0 :
                objecter1[:,0] -= shift[0]
                objecter1[:,2] -= shift[1]

                taken_place[f"{object1_name}"] = [ np.min(objecter1[:,0]), np.max(objecter1[:,0]) ,np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1]) ]
                #print("success", position)
                #print("-------------------")
                return objecter1
        elif position ==2: # case right
            #print("POS", position)
            
            min_x = random.uniform(max_x2 + outsider , max_x2 + ranger)
            
            if  min_x + (max_x1- min_x1) >= boundary_x:
                #print("no")
                continue
            
            y       = random.uniform(min_y2 + ranger2, max_y2 - ranger2)
            
            center  = ( min_x + min_x + (max_x1- min_x1)) /2 
            shift   = np.array([mean_x1, mean_y1]) - np.array([center, y])
            
            
            count   = 0

            for quarad in taken_place.values():
                if ( center + size_ob_x >= quarad[0] + overlap and  center - size_ob_x <= quarad[1] - overlap)  and (y + size_ob_y >= quarad[2] + overlap  and  y - size_ob_y <= quarad[3] - overlap) and ( z + size_ob_z >= quarad[4] + overlap and  z - size_ob_z <= quarad[5] - overlap):
                    #print("eror")
                    #print("taken", quarad, taken_place)
                    #print("center xy", center, y)
                    break
                count +=1
            #print("count", count, len(taken_place.values()))
            if count == len(taken_place.values()) and count !=0 :
                objecter1[:,0] -= shift[0]
                objecter1[:,2] -= shift[1]

                taken_place[f"{object1_name}"] = [ np.min(objecter1[:,0]), np.max(objecter1[:,0]) ,np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1]) ]
                #print("success", position)
                #print("-------------------")
                return objecter1
        elif position ==3: # case infronof
            #print("POS", position)
            min_y   = random.uniform(max_y2 + outsider, max_y2 + ranger)
            
            if  min_y + (max_y1- min_y1) >= boundary_y:
                #print("no")
                continue
            
            x       = random.uniform(min_x2 + ranger2, max_x2 - ranger2)
            
            center  = ( min_y + min_y + (max_y1- min_y1)) /2 
            shift   = np.array([mean_x1, mean_y1]) - np.array([x, center])
            
            
            count   = 0

            for quarad in taken_place.values():
                if ( center + size_ob_y >= quarad[2] + overlap and center - size_ob_y <= quarad[3] - overlap) and (x + size_ob_x >= quarad[0] + overlap and x - size_ob_x <= quarad[1] - overlap) and ( z + size_ob_z >= quarad[4] + overlap and  z - size_ob_z <= quarad[5] - overlap):
                    #print("eror")
                    #print("taken", quarad, taken_place)
                    #print("center xy",  x, center)
                    break

                count +=1
            #print("count", count, len(taken_place.values()))
            if count == len(taken_place.values()) and count !=0 :
                objecter1[:,0] -= shift[0]
                objecter1[:,2] -= shift[1]
    
                taken_place[f"{object1_name}"] = [ np.min(objecter1[:,0]), np.max(objecter1[:,0]) ,np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1]) ]
                #print("success", position)
                #print("-------------------")
                return objecter1
        elif position ==4: # case behind
            #print("POS", position)
            max_y = random.uniform(min_y2 - ranger, min_y2 - outsider)
            
            if  max_y - (max_y1- min_y1) <=0:
                #print("no")
                continue
            
            x       = random.uniform(min_x2 + ranger2, max_x2 - ranger2)
            
            center  = ( max_y + max_y - (max_y1- min_y1)) /2 
            shift   = np.array([mean_x1, mean_y1]) - np.array([x, center])
            
            count   = 0

            for quarad in taken_place.values():
                if ( center + size_ob_y >= quarad[2] + overlap and  center - size_ob_y <= quarad[3] - overlap)  and (x + size_ob_x >= quarad[0] + overlap  and  x - size_ob_x <= quarad[1] - overlap) and ( z + size_ob_z >= quarad[4] + overlap and  z - size_ob_z <= quarad[5] - overlap):
                    #print("eror")
                    #print("taken", quarad, taken_place)
                    #print("center xy",  x, center)
                    break
                count +=1
            #print("count", count, len(taken_place.values()))
            if count == len(taken_place.values()) and count !=0 :
                objecter1[:,0] -= shift[0]
                objecter1[:,2] -= shift[1]
                
                taken_place[f"{object1_name}"] = [ np.min(objecter1[:,0]), np.max(objecter1[:,0]) ,np.min(objecter1[:,2]), np.max(objecter1[:,2]), np.min(objecter1[:,1]), np.max(objecter1[:,1]) ]
                #print("success", position)
                #print("-------------------")
                return objecter1
        #print("-------------------")
    return objecter1
    



def random_location_onfloor(objecter, min_x, max_x, min_y, max_y, object_name):
    
    objecter = random_rotation(objecter)
    
    objecter[:,1] -= np.min(objecter[:,1]) - 0.1
    
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1, mean_x1, mean_y1 = np.min(objecter[:,0]), np.max(objecter[:,0]) ,np.min(objecter[:,2]), np.max(objecter[:,2]), np.min(objecter[:,1]), np.max(objecter[:,1]), np.mean(objecter[:,0]), np.mean(objecter[:,2])
    
    size_ob_x = (max_x1 - min_x1) / 2
    size_ob_y = (max_y1 - min_y1) / 2
    size_ob_z = (max_z1 - min_z1) / 2
    
    outsider = 0.1
    
    random_center_x = np.linspace(min_x + size_ob_x + outsider, max_x - size_ob_x - outsider, num=30)
    
    random_center_y = np.linspace(min_y + size_ob_y + outsider, max_y - size_ob_y - outsider, num=30)
    
    np.random.shuffle(random_center_x)
    np.random.shuffle(random_center_y)
    
    concat_xy = np.array([[a,b] for a in random_center_x for b in random_center_y ])
    
    z =  (max_z1 + min_z1)/2
    for pair in concat_xy:
        x = pair[0]
        y = pair[1]
        if taken_place == {}:
            objecter[:,0] = objecter[:,0] - mean_x1 + x
            objecter[:,2] = objecter[:,2] - mean_y1 + y

            taken_place[f"{object_name}"] = [ np.min(objecter[:,0]), np.max(objecter[:,0]) ,np.min(objecter[:,2]), np.max(objecter[:,2]), min_z1, max_z1 ]

            return objecter

        count = 0
        overlap = -0.02
        for quarad in taken_place.values():
            if ( x + size_ob_x >= quarad[0] + overlap and x - size_ob_x <= quarad[1] - overlap) and (y + size_ob_y >= quarad[2] + overlap and y - size_ob_y <= quarad[3] - overlap) and ( z + size_ob_z >= quarad[4] + overlap and  z - size_ob_z <= quarad[5] - overlap):
                break

            count +=1
        if count == len(taken_place.values()) and count !=0 :
            objecter[:,0] = objecter[:,0] - mean_x1 + x
            objecter[:,2] = objecter[:,2] - mean_y1 + y
    
            taken_place[f"{object_name}"] = [ np.min(objecter[:,0]), np.max(objecter[:,0]) ,np.min(objecter[:,2]), np.max(objecter[:,2]), min_z1, max_z1 ]
            return objecter
    
    
    return objecter
    
    



def rotate_object(objecter, degree): # rotate object with specified degree
    
    temp_object = np.zeros(np.shape(objecter))
    temp_object[:,0] = objecter[:,0]
    temp_object[:,1] = objecter[:,2]
    temp_object[:,2] = objecter[:,1]
    
    rotation_matrix = np.array([ [np.cos(degree), -np.sin(degree), 0], [np.sin(degree),  np.cos(degree),  0], [0, 0, 1],])
    rotation = np.dot(temp_object, rotation_matrix)
    
    objecter[:,0] = rotation[:,0]
    objecter[:,1] = rotation[:,2]
    objecter[:,2] = rotation[:,1]
    
    return objecter

def random_rotation(objecter): # random rotate
    numbers         = [1, 2, 2/3 , 0.5]
    random_degree   = np.pi/np.random.choice(numbers)
    return rotate_object(objecter, random_degree)


def random_object(label): # random choice object in shapenet
    if label == "table":
        point_paths = glob('data/ShapeNet55-34/ShapeNet55/shapenet_pc/04379243*.npy')
        objecter = np.load(np.random.choice(point_paths)) #* random.uniform(1.3,1.6)
    elif label == "chair":
        point_paths = glob('data/ShapeNet55-34/ShapeNet55/shapenet_pc/03001627*.npy')
        objecter = np.load(np.random.choice(point_paths)) #* random.uniform(1, 1.2)
    elif label == "printer":
        point_paths = glob('data/ShapeNet55-34/ShapeNet55/shapenet_pc/04004475*.npy')
        objecter = np.load(np.random.choice(point_paths)) #* random.uniform(0.4, 0.55)
    elif label == "laptop":
        point_paths = glob('data/ShapeNet55-34/ShapeNet55/shapenet_pc/03642806*.npy')
        objecter = np.load(np.random.choice(point_paths)) #* random.uniform(0.4, 0.55)
    elif label == "sofa":
        point_paths = glob('data/ShapeNet55-34/ShapeNet55/shapenet_pc/04256520*.npy')
        objecter = np.load(np.random.choice(point_paths)) #* random.uniform(1.2, 1.5)
    elif label == "bed":
        point_paths = glob('data/ShapeNet55-34/ShapeNet55/shapenet_pc/02818832*.npy')
        objecter = np.load(np.random.choice(point_paths)) #* random.uniform(1.4, 1.8)
    elif label == "bookshelf":
        point_paths = glob('data/ShapeNet55-34/ShapeNet55/shapenet_pc/02871439*.npy')
        objecter = np.load(np.random.choice(point_paths)) #* random.uniform(1.3,1.6)
    return random_sample(objecter, 8192)




def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
    seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    
    if isinstance(crop,list):
        num_crop = random.randint(crop[0],crop[1])   # random from 25% to 75% 
    else:
        num_crop = crop

    if fixed_points is None:       
        center = F.normalize(torch.randn(1,1,3),p=2,dim=-1)
    else:
        if isinstance(fixed_points,list):
            fixed_point = random.sample(fixed_points,1)[0]
        else:
            fixed_point = fixed_points
        center = fixed_point
    distance_matrix = torch.norm(center - xyz, p =2 ,dim = -1)  # 1 1 2048
    
    idx = torch.argsort(distance_matrix, descending=True) # 2048

    input_data = xyz.copy()
    input_data = xyz[idx[num_crop:]]

    if isinstance(crop,list):
        input_data = random_sample(input_data,2048)

    else:
        input_data = input_data

    return input_data


def random_sample(data, number):
    '''
        data B N 3
        number int
    '''
    random_indices = np.random.permutation(data.shape[0])
    # Select the desired number of points from the shuffled tensor using indexing
    sampled_points = data[random_indices[:number]]
    return sampled_points



def mainer(location):
    # create object
    table1 = random_object("table")
    table2 = random_object("table")
    table3 = random_object("table")
    
    chair1= random_object("chair")
    chair2= random_object("chair")
    chair3= random_object("chair")
    #chair4 = random_object("chair")

    # printer1 = random_object("printer")
    # laptop1 = random_object("laptop")
    sofa1 = random_object("sofa")
    sofa2 = random_object("sofa")
    sofa3 = random_object("sofa")
    
    bed1 = random_object("bed")
    bed2 = random_object("bed")
    bed3 = random_object("bed")
    

    bookshelf1 = random_object("bookshelf")
    bookshelf2 = random_object("bookshelf")
    bookshelf3 = random_object("bookshelf")
    
    #pillow1 = np.load('03938244/_/train/3fab1dacfa43a7046163a609fcf6c52.npy')
    #pillow2 = np.load('03938244/_/train/7c94c3ede6a3d3b6d0a29518c983e8c6.npy')
    #pillow3 = np.load('03938244/_/train/3fab1dacfa43a7046163a609fcf6c52.npy')
    ##pillow4 = np.load('03938244/_/train/7c94c3ede6a3d3b6d0a29518c983e8c6.npy')
    #pillow5 = np.load('03938244/_/train/3fab1dacfa43a7046163a609fcf6c52.npy')
    
    #init some 
    max_x           = np.random.randint(4,6)
    min_x           = 0
    max_y           = np.random.randint(4,6)
    min_y           = 0
    max_z           = np.random.randint(3,5)
    min_z           = 0

    thick           = 0.03
    point_thick     = 1
    fix_point       = 8196
    
    #load into dict
    
    object_dict['table1']   = table1
    object_dict['table2']   = table2
    object_dict['table3']   = table3
    object_dict['chair1']   = chair1
    object_dict['chair2']   = chair2
    object_dict['chair3']   = chair3

    # object_dict['printer1'] = printer1
    # object_dict['laptop1']  = laptop1
    object_dict['bed1']     = bed1
    object_dict['bed2']     = bed2
    object_dict['bed3']     = bed3

    object_dict['sofa1']    = sofa1
    object_dict['sofa2']    = sofa2
    object_dict['sofa3']    = sofa3
    # object_dict['pillow1']  = pillow1
    # object_dict['pillow2']  = pillow2
    # object_dict['pillow3']  = pillow3
    # object_dict['pillow4']  = pillow4
    # object_dict['pillow5']  = pillow5
    #object_dict['sofa2']    = sofa2
    #object_dict['door1']    = door1
    object_dict['bookshelf1']    = bookshelf1
    object_dict['bookshelf2']    = bookshelf2
    object_dict['bookshelf3']    = bookshelf3
    #object_dict['bookshelf4']    = bookshelf4

    # object_dict['wall']    = wall
    # object_dict['floor1']    = floor1
    # object_dict['celling1']    = celling1
    # if len(column_list) !=0:
    #     object_dict['column']    = np.concatenate(tuple(column_list), axis =0)
        
    # if len(beam_list) !=0:
    #     object_dict['beam']    = np.concatenate(tuple(beam_list), axis =0)
    # object_dict['wall_x2']    = wall_x2
    # object_dict['wall_y1']    = wall_y1
    # object_dict['wall_y2']    = wall_y2

    rules = sorted(["1_table3_on_floor",  "1_bookshelf2_on_floor","1_bookshelf3_on_floor","1_table1_on_floor", "1_table2_on_floor", "1_chair1_on_floor", "1_chair2_on_floor", "1_chair3_on_floor", 
    "2_chair1_nextto_table1", "2_chair2_nextto_table2", "2_chair3_nextto_table1", 
    "1_sofa1_on_floor", "1_bed1_on_floor", "1_bed2_on_floor","1_bed3_on_floor", "1_sofa2_on_floor",    "1_bookshelf1_on_floor", "1_sofa3_on_floor"])
    
    for rule in rules:
        spliter = rule.split("_")
        object1, object2 = spliter[1], spliter[3]
        #relation = spliter[2]
        if spliter[0] == "1": # object1 on floor
            
            object_dict[f"{object1}"][:,1] -= min(object_dict[f"{object1}"][:,1]) # -= min to min =0
            
            object_dict[f"{object1}"] = random_location_onfloor(object_dict[f"{object1}"],0,max_x,0, max_y,object1)
            
        elif spliter[0] == "2": # object1 next to object 2
            
            object_dict[f"{object1}"] = random_nextto(object_dict[f"{object1}"],object_dict[f"{object2}"],object1,object2, max_x, max_y)
            
        elif spliter[0] == "3" : # object1 on object 2
            
            object_dict[f"{object1}"] = random_on(object_dict[f"{object1}"],object_dict[f"{object2}"],object1,object2)
            
    
    totaler = np.concatenate(tuple(object_dict.values()), axis = 0)
    
    colour = [torch.ones(torch.tensor(object_dict[f"{list(object_dict.keys())[x]}"]).shape[0]).int()*(x) for x in range(len(object_dict.keys()))]

    color = torch.cat(tuple(colour), dim = 0)

    GeometricTools.drawPointCloudsColorsClasses( torch.tensor(totaler),  color, [[0,0,0]])
    
    create_folder(f"{location}/full")
    
    
    for key in object_dict.keys():
        destination = f"{key}"
        save_path = f"{location}/full/{destination}.npy"
        np.save(save_path, object_dict[key])
        
        # create 5 view point ( 4 corners and center)

        # choice_string = ["corner_0", "corner_1", "corner_2", "corner_3", "center"]
        # choice = [torch.Tensor([max_x/2,max_y,1]),torch.Tensor([max_x/2, 0, 1]),torch.Tensor([0, max_y/2, 1]),torch.Tensor([max_x, max_y/2, 1])]

        choice_string = ["corner_0", "corner_1"]
        choice = [torch.Tensor([max_x/2,max_y,0.1]),torch.Tensor([max_x/2, 0, 0.1])]

        num_crop = [int(8192 * 1/4), int(8192 * 1/2)]
        num_crop_string = ["easy", "medium"]
        create_folder(f"{location}/partial")
        for i in range(len(num_crop)): # mode ( easy, medium, hard)
            create_folder(f"{location}/partial/{num_crop_string[i]}")
            for item in range(len(choice)):
                create_folder(f"{location}/partial/{num_crop_string[i]}/{choice_string[item]}")
                
                partial = seprate_point_cloud(object_dict[key], 8192, num_crop[i], fixed_points = choice[item])
                save_path_1 = f"{location}/partial/{num_crop_string[i]}/{choice_string[item]}/{destination}.npy"
                np.save(save_path_1, partial)
            
            
def create_folder(location):
    try:
        os.mkdir(location)
        #("Folder created successfully!")
    except FileExistsError:
        a = 1
        #print("Folder already exists!")
    

def main():
    num_iter = 1
    create_folder("demo/storage")
    # main_location = "storage"


    for i in range(num_iter):
        create_folder(f"demo/storage/room_{i}")
        
        create_folder(f"demo/storage/room_{i}/full")
        
        mainer(f"demo/storage/room_{i}")  # create full



if __name__ == '__main__':
    main()
