from helper import *
import numpy as np
#from load_point_clouds import *
import os


def statistic_outlier_removal(pcd, nb_neighbors=10, std_ratio=1.0):
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                    std_ratio=std_ratio)
    
    display_inlier_outlier(pcd, ind)
    inlier_cloud = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([inlier_cloud], "Statistical oulier removal")
    return inlier_cloud


def radius_outlier_removal(pcd, nb_points=16, radius=0.05):
    _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    display_inlier_outlier(pcd, ind)
    inlier_cloud = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([inlier_cloud], "Radius oulier removal")
    return inlier_cloud


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], "Showing outliers (red) and inliers (gray)")



# if __name__ == "__main__":

#     path            = f'data/noise_removal/noise_bunny.ply'
#     if not os.path.exists(path):
#         create_noise_bunny()
#     pcd             = o3d.io.read_point_cloud(path)
#     o3d.visualization.draw_geometries([pcd], "Noise point cloud")
#     inlier_cloud    = radius_outlier_removal(pcd,5, 0.01)
#     inlier_cloud    = statistic_outlier_removal(pcd, 5, 0.5)
