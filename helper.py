
import open3d as o3d
import numpy as np
import seaborn as sns
import copy
from glob import glob 
import os
import matplotlib.pyplot as plt
import sys
import math as m

def draw_point_clouds(array, color = None, logits_color = False, name = "Open3D"):
    """
    Visualize 3D point clouds from an 3D point cloud array and its corresponding color array.
    Args:
        array (np.array): a point clouds (N,3)
        color (np.array, optional): color of the points. Defaults to None.
        logits_color (bool, optional): flags if it is True, size of input color must be (N,3). Otherwise, size of color must be (N,1). Defaults to False.
        name (str, optional): name of display window. Defaults to "Open3D".
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    if color is not None:
        if logits_color == False:
            colorLength    = abs(np.max(color)) + 1
            colorPalette   = sns.color_palette("Paired", int(colorLength))
            colorArray     = np.array(colorPalette)
            color          = colorArray[color]
            pcd.colors     = o3d.utility.Vector3dVector(color)
        else:
            if np.max(color) > 1:
                color = color/255.0
            pcd.colors = o3d.utility.Vector3dVector(color)
    
    o3d.visualization.draw_geometries([pcd], window_name = name)
    
def create_noise_bunny():
    path        = f'data/down_sampling/bun_zipper.ply'
    pcd1        = o3d.io.read_point_cloud(path)
    
    p1_load     = np.asarray(pcd1.points)
    p1_color    = np.zeros((p1_load.shape[0],3)) + np.array([0.8,0.8,0.8]) 
    p1_norm     = (p1_load - np.min(p1_load))/ (np.max(p1_load) - np.min(p1_load))
    a           = np.random.rand(1000,3)/2 + (np.min(p1_norm, axis = 0) - 0.05)
    a_colors    =  np.zeros((a.shape[0],3)) + np.array([1,0.5,1])
    p3_load     = np.concatenate((p1_norm,a), axis=0)
    p3_color    = np.concatenate((p1_color,a_colors), axis=0)
    pcd         = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(p3_load)
    pcd.colors  = o3d.utility.Vector3dVector(p3_color)
    o3d.io.write_point_cloud('data/noise_removal/noise_bunny.ply',pcd)
    
def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                        node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


def draw_registration_result(source, target, transformation, name):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name = name)
    
    
def show_rgbd(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('Grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()