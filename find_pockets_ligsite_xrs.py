import numpy as np
import sys
import math
from collections import defaultdict


van_der_Waals_radii = {
    "C": 1.70,
    "H": 1.17,
    "O": 1.40,
    "N": 1.50,
    "S": 1.85
}

r_atom_max = max([radius for radius in van_der_Waals_radii.values()])
probe_radius = 1.4

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

def atoms_coordinates_dict(file_path):
    atom_id_coords_dict = {}
    for atom_id, _, _, _, _, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        atom_id_coords_dict[atom_id] = (x_coordinate, y_coordinate, z_coordinate)
    return atom_id_coords_dict
print(atoms_coordinates_dict("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1mh1.pdb"))

def get_atoms_and_residues(pdb_file_path):
    """Extract atoms and residues from PDB file."""
    atoms_coords = []
    atom_id_coords_dict = {}
    residues = {}
    
    for atom_id, atom_name, residue_name, chain_ID, residue_ID, x, y, z in PDB_iterator(pdb_file_path):
        atom_id_coords_dict[atom_id] = (x, y, z)
        atoms_coords.append((x, y, z))
        
        # Store residue information
        residue_key = (chain_ID, residue_ID)
        if residue_key not in residues:
            residues[residue_key] = {
                'name': residue_name,
                'atoms': []
            }
        residues[residue_key]['atoms'].append((atom_id, atom_name, (x, y, z)))
    
    return np.array(atoms_coords), atom_id_coords_dict, residues
    #return atom_id_coords_dict
print(get_atoms_and_residues("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1mh1.pdb"))
    
# The next step is to divide the 3D space (the bounding box in this case) into small cubes of x Angstrom per side
def create_bounding_box_and_voxels(atom_coordinates, voxel_size = 1.0):

    x_coords = []
    y_coords = []
    z_coords = []

    for _, coordinates in atom_coordinates.items():
            x_coords.append(coordinates[0])
            y_coords.append(coordinates[1])
            z_coords.append(coordinates[2])
    
   
    r_max = r_atom_max + probe_radius

    xmin = min(x_coords) - r_max
    xmax = max(x_coords) + r_max
    ymin = min(y_coords) - r_max
    ymax = max(y_coords) + r_max
    zmin = min(z_coords) - r_max
    zmax = max(z_coords) + r_max

    bounding_box_dict = {"X":[xmin, xmax], "Y":[ymin, ymax], "Z":[zmin, zmax]}

    # Add 1 to ensure we cover the bounding box
    # TODO: is 1 or voxel_size what we need to add??
    range_x = math.floor((xmax - xmin)/voxel_size) + 1
    range_y = math.floor((ymax - ymin)/voxel_size) + 1
    range_z = math.floor((zmax - zmin)/voxel_size) + 1
    #print(f"Voxel grid dimensions are: rangex={range_x}, rangey={range_y}, rangez={range_z}")

    # Generate voxel coordinates and store them in a dictionary
    voxel_grid = {}
    for i in range(range_x): # recall: range(n) = 0, 1, 2, ..., n-1
        for j in range(range_y):
            for k in range(range_z):
                #Store in a dictionary with the indices i,j,k as key
                voxel_grid[(i,j,k)] = 0
    
    return bounding_box_dict, voxel_grid


# For atom in PDB determine all coordinates of voxels that overlap with this atom
# set the grid coordinates to -1 if that’s the case 
def determine_if_within_bounding_cube_for_atom(center, point):
    # Compute Euclidean distance from atom center to voxel center
    distance = np.linalg.norm(center - point)
    if distance <= van_der_Waals_radii[center]:
        return True


def mark_occupied_voxels(atom_coordinates, file_path, voxel_size = 1.0):
  
    box, voxel_grid = create_bounding_box_and_voxels(atom_coordinates, voxel_size)

    xmin = box["X"][0]
    ymin = box["Y"][0]
    zmin = box["Z"][0]
    
    for idx, (_, atom_name, _, _, _, x, y, z) in enumerate(PDB_iterator(file_path)):
        element = atom_name[0]
        r_atom = van_der_Waals_radii[element]
        r = r_atom + probe_radius

        # Compute voxel index in each direction
        # Convert atom's "bounding box" coordinates into voxel grid indices
        i0 = math.floor((x - r - xmin) / voxel_size)
        j0 = math.floor((y - r - ymin) / voxel_size)
        k0 = math.floor((z - r - zmin) / voxel_size)
        i1 = math.floor((x + r - xmin) / voxel_size) + 1
        j1 = math.floor((y + r - ymin) / voxel_size) + 1
        k1 = math.floor((z + r - zmin) / voxel_size) + 1

        # Determine which voxel this atom is in
        #i = math.floor((x - xmin) / voxel_size) # because indices start at 0
        #j = math.floor((y - ymin) / voxel_size)
        #k = math.floor((z - zmin) / voxel_size)
        
        # Iterate over the voxel range defined above
        for l in range(i0, i1 + 1):
            for m in range(j0, j1 + 1):
                for n in range(k0, k1 + 1):
                    # Convert voxel index to Cartesian coordinate
                    voxel_x = xmin + l*voxel_size
                    voxel_y = ymin + m*voxel_size
                    voxel_z = zmin + n*voxel_size

                    # Compute distance to atom center
                    distance = math.sqrt((voxel_x - x)**2 + (voxel_y - y)**2 + (voxel_z - z)**2)
                   
                    if distance <= r:
                        if (l, m, n) in voxel_grid:
                            # Mark voxel as-1, i.e.occupied
                            voxel_grid[(i,j,k)] = -1
    
    return voxel_grid

def identify_filled_empty_voxels(file_path, voxel_size=1.0):
    """
    Step 4: Separate empty voxels from voxels filled by protein atoms in the convex hull
    
    Returns:
    - voxel_grid: Dictionary with voxel indices as keys and 'filled' or 'empty' as values
    - atom_voxel_map: Dictionary mapping each atom index to its voxel
    - voxel_atom_map: Dictionary mapping each filled voxel to the atoms it contains
    """
    box_coordinates = define_bounding_box(file_path)
    xmin, xmax = box_coordinates["X"]
    ymin, ymax = box_coordinates["Y"]
    zmin, zmax = box_coordinates["Z"]
    
    # Calculate grid dimensions
    range_x = math.ceil((xmax - xmin)/voxel_size)
    range_y = math.ceil((ymax - ymin)/voxel_size)
    range_z = math.ceil((zmax - zmin)/voxel_size)
    
    # Initialize voxel grid
    voxel_grid = {}
    for i in range(range_x):
        for j in range(range_y):
            for k in range(range_z):
                voxel_grid[(i, j, k)] = 'empty'
    
    # Map atoms to voxels
    atom_voxel_map = {}
    voxel_atom_map = defaultdict(list)
    
    for idx, (identifier, atom_name, _, _, _, x, y, z) in enumerate(PDB_iterator(file_path)):
        # Determine which voxel this atom is in
        i = int((x - xmin) / voxel_size)
        j = int((y - ymin) / voxel_size)
        k = int((z - zmin) / voxel_size)
        
        voxel_idx = (i, j, k)
        voxel_grid[voxel_idx] = 'filled'
        atom_voxel_map[identifier] = voxel_idx
        voxel_atom_map[voxel_idx].append(identifier)
    
    # Get convex hull to determine which empty voxels are inside
    atoms_list = []
    for _, _, _, _, _, x, y, z in PDB_iterator(file_path):
        atoms_list.append((x, y, z))
    
    hull = ConvexHull(np.array(atoms_list))
    
    return voxel_grid, atom_voxel_map, voxel_atom_map, hull
