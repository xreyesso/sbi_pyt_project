import numpy as np
import sys
import math
from collections import defaultdict

# Step 0: Read PDB file and get atom coordinates
# Step 1: create 3D grid for the protein
# Step 2: Set a value for each voxel. We start by setting all voxels to 0
# a voxel that is inaccessible to solvent (because it is already occupied by the protein) gets a value of -1
# Step 3: Scan along the x, y and z axis to detect solvent voxels enclosed on both sides by -1
# Step 4: Scan along 4 cubic diagonals
# A sequence of voxels which starts with protein, followed by solvent and ending with protein is 
# calleda protein-solvent-protein (PSP) event.
# Step 5: Detect pockets as regions of voxels with a minimum number of PSP events (MIN_PSP)
# Check nearest neighbors
# STEP 6
# Distinguish cavities
# STEP 7
# Determine the surface of a pocket
# STEP 8
# Identify the aminoacids and atoms that surround the surface of a pocket
# STEP 9
# Prepare output and display

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
#print(atoms_coordinates_dict("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1mh1.pdb"))

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
#print(get_atoms_and_residues("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1mh1.pdb"))

# STEP 1    
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
    # TODO: is 1 or voxel_size what we need to add?? It's 1, it's indices not cartesian coordinates, voxel_size is the wrong unit
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
    
    return bounding_box_dict, voxel_grid, (range_x, range_y, range_z)


# For atom in PDB determine all coordinates of voxels that overlap with this atom
# set the grid coordinates to -1 if that’s the case 
# def determine_if_within_bounding_cube_for_atom(center, point):
#     # Compute Euclidean distance from atom center to voxel center
#     distance = np.linalg.norm(center - point)
#     if distance <= van_der_Waals_radii[center]:
#         return True

# STEP 2
def mark_occupied_voxels(atom_coordinates, file_path, voxel_size = 1.0):
  
    box, voxel_grid, _ = create_bounding_box_and_voxels(atom_coordinates, voxel_size)

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
                    # Convert voxel index to Cartesian coordinate and add voxel_size/2 to each 
                    # coordinate to get the voxel center
                    voxel_x = xmin + l*voxel_size + voxel_size / 2
                    voxel_y = ymin + m*voxel_size + voxel_size / 2
                    voxel_z = zmin + n*voxel_size + voxel_size / 2

                    # Compute distance from voxel center to atom center
                    distance = math.sqrt((voxel_x - x)**2 + (voxel_y - y)**2 + (voxel_z - z)**2)
                   
                    if distance <= r:
                        if (l, m, n) in voxel_grid:
                            # Mark voxel as-1, i.e.occupied
                            voxel_grid[(l,m,n)] = -1
    
    return voxel_grid

# STEP 3
# Find solvent-accessible voxels (value = 0) that are enclosed between inaccessible voxels (value = -1)
# along straight lines
# We are looking for runs of zeros that are enclosed on both sides by -1, only those get incremented by 1
def mark_enclosed_voxels(scanline, index_map, voxel_grid):
    pass

def scan_along_axis(atom_coordinates, voxel_grid, axis):
    _, _, (range_x, range_y, range_z) = create_bounding_box_and_voxels(atom_coordinates)

    if axis == 'x':
        scan_range = range(range_x)
        dim1_range = range(range_y)
        dim2_range = range(range_z)
    elif axis == 'y':
        dim1_range = range(range_x)
        scan_range = range(range_y)
        dim2_range = range(range_z)
    elif axis == 'z':
        dim1_range = range(range_x)
        dim2_range = range(range_y)
        scan_range = range(range_z)
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")


    for i in dim1_range:
        for j in dim2_range: # for all lines parallel to the indicated axis
            solvent_voxels = None

            for s in scan_range:
                if axis == 'x':
                    key = (s, i, j)
                elif axis == 'y':
                    key = (i, s, j)
                else:
                    key = (i, j, s)

                if voxel_grid[key] == -1:
                    if solvent_voxels is not None:
                        for key2 in solvent_voxels:
                            voxel_grid[key2] += 1 # Add 1 for the solvent voxels in between protein voxels
                    solvent_voxels = [] # Reset the solvent voxels list so we do not process them multiple times
                else:
                    if solvent_voxels is not None:
                        solvent_voxels.append(key) # Only keep track of solvent voxels if they come after a protein (occupied) voxel
    
    return(voxel_grid)
                
# STEP 4
# Scan along the diagonals
def scan_along_diagonal(atom_coordinates, voxel_grid, diagonal_vector):
    """
    Scan along a diagonal direction specified by diagonal_vector.
    diagonal_vector should be one of:
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)
    """
    _, _, (range_x, range_y, range_z) = create_bounding_box_and_voxels(atom_coordinates)
    
    # Unpack diagonal direction
    dx, dy, dz = diagonal_vector
    
    # Determine starting points based on the direction
    # We need to consider all possible starting points for the diagonal
    if dx > 0:
        x_starts = range(range_x)
    else:
        x_starts = range(range_x - 1, -1, -1) # Start from the end
    if dy > 0:
        y_starts = range(range_y)
    else:
        y_starts = range(range_y - 1, -1, -1) # Start from the end
    if dz > 0:
        z_starts = range(range_z)
    else:
        z_starts = range(range_z - 1, -1, -1) # Start from the end

    # Plane (x,y) scan
    for start_x in x_starts:
        for start_y in y_starts:
            # Start a diagonal trace from this point
            x, y, z = start_x, start_y, 0
            solvent_voxels = None
            occupied_voxel = False
            
            # Follow the diagonal until we go out of bounds
            while 0 <= x < range_x and 0 <= y < range_y and 0 <= z < range_z:
                key = (x, y, z)
                
                if voxel_grid[key] == -1:  # Protein voxel
                    occupied_voxel = True
                    if solvent_voxels is not None:
                        for key2 in solvent_voxels:
                            voxel_grid[key2] += 1  # Increment solvent voxels between protein voxels
                    solvent_voxels = []  # Reset for next potential cavity
                else:  # Solvent voxel
                    if occupied_voxel:
                        solvent_voxels.append(key)
                
                # Move along the diagonal
                x += dx
                y += dy
                z += dz

    # Repeat the process for all z-starting positions
    for start_x in x_starts:
        for start_z in z_starts:
            # Start a diagonal trace from this point
            x, y, z = start_x, 0, start_z
            solvent_voxels = None
            occupied_voxel = False
            
            # Follow the diagonal until we go out of bounds
            while 0 <= x < range_x and 0 <= y < range_y and 0 <= z < range_z:
                key = (x, y, z)
                
                if voxel_grid.get(key, 0) == -1:  # Protein voxel
                    occupied_voxel = True
                    if solvent_voxels is not None:
                        for key2 in solvent_voxels:
                            voxel_grid[key2] += 1  # Increment solvent voxels between protein voxels
                    solvent_voxels = []  # Reset for next potential cavity
                else:  # Solvent voxel
                    if occupied_voxel:
                        solvent_voxels.append(key)
                
                # Move along the diagonal
                x += dx
                y += dy
                z += dz

    # Repeat the process for all y-starting positions
    for start_y in y_starts:
        for start_z in z_starts:
            # Start a diagonal trace from this point
            x, y, z = 0, start_y, start_z
            solvent_voxels = None
            occupied_voxel = False
            
            # Follow the diagonal until we go out of bounds
            while 0 <= x < range_x and 0 <= y < range_y and 0 <= z < range_z:
                key = (x, y, z)
                
                if voxel_grid.get(key, 0) == -1:  # Protein voxel
                    occupied_voxel = True
                    if solvent_voxels is not None:
                        for key2 in solvent_voxels:
                            voxel_grid[key2] += 1  # Increment solvent voxels between protein voxels
                    solvent_voxels = []  # Reset for next potential cavity
                else:  # Solvent voxel
                    if occupied_voxel:
                        solvent_voxels.append(key)
                
                # Move along the diagonal
                x += dx
                y += dy
                z += dz

    return(voxel_grid)

# STEP 5
# Define pockets and cavities
def define_pockets_and_cavities(voxel_grid, MIN_PSP = 2):
    # Start by finding a voxel whose value is >= MIN_PSP
    # Add to the region all nearest neighbors (how do we define this?) with values >= MIN_PSP
    # Newly added nearest neighbors are again checked for nearest neighbors with values >= MIN_PSP
    # Continue until all nearest neighbors with values >= MIN_PSP are added to the region
    # Any voxels left with values >= MIN_PSP constitute one or more new pockets, and we start the process again for them

    # return: a list of pockets? or a dictionary?
    pass

# STEP 6
# Distinguish cavities
# TODO: Is this necessary?
def distinguish_cavities(voxel_grid, MIN_PSP = 2):

    pass

# STEP 7
# Determine the surface of a pocket
def determine_pocket_surface():
    # input: the list/dictionary pockets from step 5
    pass

# STEP 8
# Identify the aminoacids and atoms that surround the surface of a pocket

# STEP 9
# Prepare output and display, for example create a PDB-like file
# Prepare a file that can be read by PyMol

def run_complete_workflow(file_path):
    
    atoms_ids_and_coordinates = atoms_coordinates_dict(file_path)

    # Create voxels
    _, voxel_grid, _ = create_bounding_box_and_voxels(atoms_ids_and_coordinates)

    voxel_grid = mark_occupied_voxels(atoms_ids_and_coordinates, file_path)

    axes = ['x','y','z']
    for axis in axes:
        voxel_grid = scan_along_axis(atoms_ids_and_coordinates, voxel_grid, axis)

    diagonals = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)]
    for diag in diagonals:
        voxel_grid = scan_along_diagonal(atoms_ids_and_coordinates, voxel_grid, diag)

    print(voxel_grid)
run_complete_workflow("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1mh1.pdb")