import torch
import numpy as np
import open3d as o3d
import seaborn as sns
import ast
from collections import deque

def farthest_point_sampling(arr, n_sample, start_idx=None):
    """Farthest Point Sampling without the need to compute all pairs of distance.

    Parameters
    ----------
    arr : numpy array
        The positional array of shape (n_points, n_dim)
    n_sample : int
        The number of points to sample.
    start_idx : int, optional
        If given, appoint the index of the starting point,
        otherwise randomly select a point as the start point.
        (default: None)

    Returns
    -------
    numpy array of shape (n_sample,)
        The sampled indices.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 1024)
    >>> point_idx = farthest_point_sampling(data, 3)
    >>> print(point_idx)
        array([80, 79, 27])

    >>> point_idx = farthest_point_sampling(data, 5, 60)
    >>> print(point_idx)
        array([60, 39, 59, 21, 73])
    """
    n_points, n_dim = arr.shape

    if (start_idx is None) or (start_idx < 0):
        start_idx = np.random.randint(0, n_points)

    sampled_indices = [start_idx]
    min_distances = np.full(n_points, np.inf)
    
    for _ in range(n_sample - 1):
        current_point = arr[sampled_indices[-1]]
        dist_to_current_point = np.linalg.norm(arr - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)

    return np.array(sampled_indices)

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc  = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

class FuncCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self._name = deque()

    @property
    def name(self):
        return '.'.join(self._name)

    @name.deleter
    def name(self):
        self._name.clear()

    def visit_Name(self, node):
        self._name.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            self._name.appendleft(node.attr)
            self._name.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)


def get_func_calls(tree):
    func_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callvisitor = FuncCallVisitor()
            callvisitor.visit(node.func)
            func_calls.append(callvisitor.name)

    return func_calls


def drawPointCloudsColorsLogits(array, colors = None, origin = None, name = "Open3D"):
    pcd = o3d.geometry.PointCloud()
    if not isinstance(array, np.ndarray):
        array = array.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(array)
    if colors is not None:
        if not isinstance(colors, np.ndarray):
            colors = colors.detach().cpu().numpy()
        if np.max(colors) > 2:
            colors = colors/255.0
            
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if origin is not None:
        origin_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.7, origin=origin)
        o3d.visualization.draw_geometries([pcd, origin_draw], window_name = name)
        return
    o3d.visualization.draw_geometries([pcd], window_name = name)
        

def drawPointCloudsColorsClasses(array, colors = None, origins = None, add_geometry = None, name = "Open3D"):
    pcd = o3d.geometry.PointCloud()
    if not isinstance(array, np.ndarray):
        array = array.detach().cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(array)
    if colors is not None: 
        if not isinstance(colors, np.ndarray):
            colors = colors.detach().cpu().numpy()
        
        colorLength    = abs(np.max(colors)) + 1
        
        colorPalette   = sns.color_palette("Paired", int(colorLength))
        colorArray     = np.array(colorPalette)
        colors         = colorArray[colors]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if add_geometry is not None and origins is not None:
        origins_draw = []
        for origin in origins:
            origin_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.7, origin=origin)
            origins_draw.append(origin_draw)
        o3d.visualization.draw_geometries([pcd, *origins_draw, add_geometry], window_name = name)
        return
    
    if origins is not None:
        origins_draw = []
        for origin in origins:
            origin_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.7, origin=origin)
            origins_draw.append(origin_draw)
        
        o3d.visualization.draw_geometries([pcd, *origins_draw], window_name = name)
        return
    o3d.visualization.draw_geometries([pcd], window_name = name)

def drawPointClouds3DBoxes(points, pointColors, boxes, boxColors, name = "Open3D"):
    listInputs = [points, pointColors, boxes, boxColors] 
    if not isinstance(points, np.ndarray):
        points = points.detach().cpu().numpy() 
    if not isinstance(pointColors, np.ndarray):
        pointColors = pointColors.detach().cpu().numpy() 
    if not isinstance(boxes, np.ndarray):
        boxes = boxes.detach().cpu().numpy() 
    if not isinstance(boxColors, np.ndarray):
        boxColors = boxColors.detach().cpu().numpy() 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pointColors)
    i = 0
    center = boxes[i, 0:3]
    dim = boxes[i, 3:6]
    yaw = np.zeros(3)
    yaw[2] = boxes[i, 6]
    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(
        box3d)
    line_set.paint_uniform_color(np.array(boxColors[i]) / 255.)
    # draw bboxes on visualizer
    o3d.visualization.draw_geometries([pcd, line_set], window_name = name)

    


    
    

def info(object_x):
    print("================= MASTER DEBUG INFO ===================")
    types = type(object_x)
    print("Type: ",types)
    flag = False
    
    if isinstance(object_x, list):
        print('Length: ', len(object_x))
        flag = True
    elif torch.is_tensor(object_x):
        print('Shape: ', object_x.shape)
        print("Min: ", torch.min(object_x))
        print("Max: ", torch.max(object_x))
        flag = True
    elif isinstance(object_x, np.ndarray):
        print('Shape: ', object_x.shape)
        print("Min: ", np.min(object_x))
        print("Max: ", np.max(object_x))
        flag = True
    elif isinstance(object_x, int):
        print("Value: ",object_x)
    if flag:
        print("Some sample: ", object_x[:2])
    print(" ========================================================== ")
    
    
def discreterization(point_clouds):
    xmin,ymin,zmin = torch.min(point_clouds[:,0]).item(),torch.min(point_clouds[:,1]).item(),torch.min(point_clouds[:,2]).item()
    xmax,ymax,zmax = torch.max(point_clouds[:,0]).item(),torch.max(point_clouds[:,1]).item(),torch.max(point_clouds[:,2]).item()
    distance_x     = (xmax - xmin)/3
    distance_y     = (ymax - ymin)/3
    distance_z     = (zmax - zmin)/3
    x_coor       = []
    y_coor       = []
    z_coor       = []
    for index in range(0,4):
        x_coor.append(xmin + index*distance_x)
        y_coor.append(ymin + index*distance_y)
        z_coor.append(zmin + index*distance_z)
    
    x_coor = np.array(x_coor)
    y_coor = np.array(y_coor)
    z_coor = np.array(z_coor)
    
    x_array, y_array, z_array = np.meshgrid(x_coor, y_coor,z_coor)
    lst_points = []
    for i_small in range(x_array.shape[0]):
        for j_small in range(x_array.shape[1]):
            for k_small in range(x_array.shape[2]): 
                lst_points.append([x_array[i_small][j_small][k_small], y_array[i_small][j_small][k_small], z_array[i_small][j_small][k_small]])
    
    connect = []          
    for i_small in range(len(lst_points)):
        for j_small in range(len(lst_points)):
            if i_small == j_small: 
                break
            count = 0
            for k_small in range(3):
                if (lst_points[i_small][k_small] == lst_points[j_small][k_small]):
                    count += 1
                if count == 2:
                    connect.append([i_small,j_small])
    return lst_points, connect

def normalize_coordinate(points_tensor):
        max_point_x    = torch.max(points_tensor[:,0])
        max_point_y    = torch.max(points_tensor[:,1])
        max_point_z    = torch.max(points_tensor[:,2])
        min_point_x    = torch.min(points_tensor[:,0])
        min_point_y    = torch.min(points_tensor[:,1])
        min_point_z    = torch.min(points_tensor[:,2])
        
        distance_x = (max_point_x - min_point_x)/2
        distance_y = (max_point_y - min_point_y)/2
        distance_z = (max_point_z - min_point_z)/2
        
        center_x   = min_point_x + distance_x
        center_y   = min_point_y + distance_y
        center_z   = min_point_z + distance_z
        
        points_tensor_norm      = points_tensor.clone()
        points_tensor_norm[:,0] = points_tensor[:,0] - center_x
        points_tensor_norm[:,1] = points_tensor[:,1] - center_y
        points_tensor_norm[:,2] = points_tensor[:,2] - center_z
        
        
        center_coor     = torch.tensor([center_x,center_y,center_z], device="cuda:0")
        min_coor        = torch.tensor([min_point_x,min_point_y,min_point_z], device = "cuda:0")
        max_coor        = torch.tensor([max_point_x, max_point_y, max_point_z], device = "cuda:0")
        return points_tensor_norm, center_coor, min_coor, max_coor


def center_object(points_tensor: [torch.Tensor]) -> [list]:
    max_point_x    = torch.max(points_tensor[:,0])
    max_point_y    = torch.max(points_tensor[:,1])
    max_point_z    = torch.max(points_tensor[:,2])
    min_point_x    = torch.min(points_tensor[:,0])
    min_point_y    = torch.min(points_tensor[:,1])
    min_point_z    = torch.min(points_tensor[:,2])
    
    distance_x = (max_point_x - min_point_x)/2
    distance_y = (max_point_y - min_point_y)/2
    distance_z = (max_point_z - min_point_z)/2
    
    center_x   = min_point_x + distance_x
    center_y   = min_point_y + distance_y
    center_z   = min_point_z + distance_z
    center_coor     = [center_x.item(),center_y.item(),center_z.item()]
    return center_coor


def center_object_tensor(points_tensor):
    
    min_point   = torch.min(points_tensor, dim= 1)[0]
    max_point   = torch.max(points_tensor, dim= 1)[0]
    distance    = (max_point - min_point)/2
    center      = min_point + distance
    
    clone_tensor = center.clone().detach().requires_grad_(False)
    return clone_tensor
    
