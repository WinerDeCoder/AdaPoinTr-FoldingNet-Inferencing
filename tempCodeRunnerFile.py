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