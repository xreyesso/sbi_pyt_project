import sys
import math
import numpy as np
from scipy.spatial import ConvexHull

def PDB_iterator(pdb_file_path=None):
    if pdb_file_path is None:
        fi = sys.stdin
    else:
        fi = open(pdb_file_path,"r")

    for line in fi:
        line = line.strip()

        record_name = line[0:6].strip()

        if (record_name == "ATOM"):
            serial_number = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_ID = line[21].strip()
            residue_ID = int(line[22:26].strip())
            x_coordinate = float(line[30:38].strip())
            y_coordinate = float(line[38:46].strip())
            z_coordinate = float(line[46:54].strip())
            yield(serial_number, atom_name, residue_name, chain_ID, residue_ID, x_coordinate, y_coordinate, z_coordinate)

    fi.close()

def define_bounding_box(file_path):
    """
    Method to determine the coordinates of the box that will enclose the protein
    """
    x_coords = []
    y_coords = []
    z_coords = []

    for identifier, atom_name, residue_name, chain_ID, residue_ID, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        x_coords.append(x_coordinate)
        y_coords.append(y_coordinate)
        z_coords.append(z_coordinate)
    
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    zmin = min(z_coords)
    zmax = max(z_coords)

    bounding_box_dict = {"X":[xmin, xmax], "Y":[ymin, ymax], "Z":[zmin, zmax]}
    return bounding_box_dict      

print(define_bounding_box("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb"))

# The net step is to divide the 3D space (the bounding box in this case) into small cubes of 1 Angstrom per 
def create_voxels(file_path):
    voxel_size = 1.0
    box_coordinates = define_bounding_box(file_path)
    xmin = box_coordinates["X"][0]
    print(f"xmin: {xmin}")
    xmax = box_coordinates["X"][1]
    print(f"xmax: {xmax}")
    ymin = box_coordinates["Y"][0]
    ymax = box_coordinates["Y"][1]
    zmin = box_coordinates["Z"][0]
    zmax = box_coordinates["Z"][1]
    print(f"zmax: {zmax}")

    # Use the ceiling function to ensure we cover the bounding box
    range_x = math.ceil((xmax - xmin )/voxel_size)
    range_y = math.ceil((ymax - ymin)/voxel_size)
    range_z = math.ceil((zmax - zmin)/voxel_size)
    print(f"Voxel grid dimensions are: rangex={range_x}, rangey={range_y}, rangez={range_z}")

    # Generate voxel coordinates and store them in a dictionary
    voxel_dict = {}
    for i in range(range_x): # recall: range(n) = 0, 1, 2, ..., n-1
        for j in range(range_y):
            for k in range(range_z):
                # Calculate the center of each voxel
                center_x = xmin + i*voxel_size + (voxel_size/2)
                center_y = ymin + j*voxel_size + (voxel_size/2)
                center_z = zmin + k*voxel_size + (voxel_size/2)
                #Store in a dictionary with the indices i,j,k as key
                voxel_dict[(i,j,k)] = (center_x, center_y, center_z)

create_voxels("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb")

# Use as points the voxel centers that contain protein atoms and apply an algorithm to find the convex hull to these points
# The convex hull is the smallest convex polyhedron that encloses the protein

# define the points
# points = np.array(voxels)
def create_convex_hull(file_path):
    atoms_list = []
    for _ , _, _, _, _, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        atoms_list.append((x_coordinate,y_coordinate,z_coordinate))
    atoms_np_array = np.array(atoms_list)
    hull = ConvexHull(atoms_np_array)
    #return hull

    import matplotlib.pyplot as plt
    import matplotlib.tri as tri


    ax = plt.figure().add_subplot(projection='3d')

    ax.plot_trisurf(
        np.array([atom[0] for atom in atoms_list]),
        np.array([atom[1] for atom in atoms_list]),
        np.array([atom[2] for atom in atoms_list]),
        triangles=np.array(hull.simplices),
        linewidth=0.2, antialiased=True)

    plt.show()

    return hull

print(create_convex_hull("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb"))


# print(points[hull.vertices])