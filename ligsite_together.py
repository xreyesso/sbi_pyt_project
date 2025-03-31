import numpy as np
import sys
import math
import os
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
def create_bounding_box_and_voxels(atom_coordinates, voxel_size = 0.5):

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
def mark_occupied_voxels(atom_coordinates, file_path, voxel_size = 0.5):
  
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
def scan_along_axis(voxel_grid, grid_dimensions, axis):
    range_x, range_y, range_z = grid_dimensions

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
    
    return voxel_grid
                
# STEP 4
# Scan along the diagonals
def scan_along_diagonal(grid_dimensions, voxel_grid, diagonal_vector):
    """
    Scan along a diagonal direction specified by diagonal_vector.
    diagonal_vector should be one of:
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)
    """
    range_x, range_y, range_z = grid_dimensions
    
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

    # # Repeat the process for all z-starting positions
    # for start_x in x_starts:
    #     for start_z in z_starts:
    #         # Start a diagonal trace from this point
    #         x, y, z = start_x, 0, start_z
    #         solvent_voxels = None
    #         occupied_voxel = False
            
    #         # Follow the diagonal until we go out of bounds
    #         while 0 <= x < range_x and 0 <= y < range_y and 0 <= z < range_z:
    #             key = (x, y, z)
                
    #             if voxel_grid.get(key, 0) == -1:  # Protein voxel
    #                 occupied_voxel = True
    #                 if solvent_voxels is not None:
    #                     for key2 in solvent_voxels:
    #                         voxel_grid[key2] += 1  # Increment solvent voxels between protein voxels
    #                 solvent_voxels = []  # Reset for next potential cavity
    #             else:  # Solvent voxel
    #                 if occupied_voxel:
    #                     solvent_voxels.append(key)
                
    #             # Move along the diagonal
    #             x += dx
    #             y += dy
    #             z += dz

    # # Repeat the process for all y-starting positions
    # for start_y in y_starts:
    #     for start_z in z_starts:
    #         # Start a diagonal trace from this point
    #         x, y, z = 0, start_y, start_z
    #         solvent_voxels = None
    #         occupied_voxel = False
            
    #         # Follow the diagonal until we go out of bounds
    #         while 0 <= x < range_x and 0 <= y < range_y and 0 <= z < range_z:
    #             key = (x, y, z)
                
    #             if voxel_grid.get(key, 0) == -1:  # Protein voxel
    #                 occupied_voxel = True
    #                 if solvent_voxels is not None:
    #                     for key2 in solvent_voxels:
    #                         voxel_grid[key2] += 1  # Increment solvent voxels between protein voxels
    #                 solvent_voxels = []  # Reset for next potential cavity
    #             else:  # Solvent voxel
    #                 if occupied_voxel:
    #                     solvent_voxels.append(key)
                
    #             # Move along the diagonal
    #             x += dx
    #             y += dy
    #             z += dz

    return(voxel_grid)


# STEP 5 
# Define pockets and cavities
# pockets is a list of lists
def define_pockets_and_cavities(voxel_grid, grid_dimensions, MIN_PSP=3):
    visited = set()
    pockets = []
    range_x, range_y, range_z = grid_dimensions

    # Find all grid points with values >= MIN_PSP
    for i in range(range_x):
        for j in range(range_y):
            for k in range(range_z):
                voxel = (i, j, k)
                
                # Skip if already visited or below threshold or occupied by protein
                if voxel in visited or voxel_grid.get(voxel, 0) < MIN_PSP:
                    continue
                
                # Start a new pocket
                pocket = []
                queue = deque([voxel])
                visited.add(voxel)
                
                # Region growing process
                while queue:
                    current = queue.popleft()
                    pocket.append(current)
                    
                    # Check 26-connected neighbors (full neighborhood)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            for dk in (-1, 0, 1):
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                
                                neighbor = (current[0]+di, current[1]+dj, current[2]+dk)
                                
                                if (neighbor not in visited and 
                                    0 <= neighbor[0] < range_x and 
                                    0 <= neighbor[1] < range_y and 
                                    0 <= neighbor[2] < range_z and
                                    voxel_grid.get(neighbor, 0) >= MIN_PSP and
                                    voxel_grid.get(neighbor, 0) != -1):
                                    
                                    visited.add(neighbor)
                                    queue.append(neighbor)
                
                # Add the completed pocket
                if pocket:
                    pockets.append(pocket)
    
    # Sort pockets by size (largest first)
    return sorted(pockets, key=len, reverse=True)

def filter_pockets_by_size(pockets, min_voxel_count=30):
    return [pocket for pocket in pockets if len(pocket) >= min_voxel_count]

# STEP 6
# Determine the surface of a pocket
def determine_pocket_surface(voxel_grid, pockets):
    """
    Determine the surface of each pocket.
    
    Surface points are grid points that:
    1. Belong to a pocket (value >= MIN_PSP)
    2. Have at least one neighboring voxel that is occupied by protein (value = -1)
    
    Parameters:
    -----------
    voxel_grid : dict
        Dictionary mapping (i,j,k) coordinates to voxel values
    pockets : dict or list
        Collection of pockets identified in Step 5
        
    Returns:
    --------
    dict
        Dictionary mapping pocket ID to a dictionary of surface points
        Each surface point maps to a list of its neighboring surface points
    """
    
    # Turn the pockets (list of lists) into a dictionary
    pockets_dict = {}
    for i, pocket in enumerate(pockets):
        pockets_dict[f'pocket_{i+1}'] = pocket
    
    pockets_surface_dict = defaultdict(list)

    # Define the 6 nearest neighbors in 3D grid (face-adjacent)
    nearest_neighbors = [
        (1, 0, 0), (-1, 0, 0),  # x-axis neighbors
        (0, 1, 0), (0, -1, 0),  # y-axis neighbors
        (0, 0, 1), (0, 0, -1)   # z-axis neighbors
    ]
    
    # Optional: include diagonal neighbors for a more detailed surface
    # This would include 20 additional neighbors (12 edge-adjacent + 8 vertex-adjacent)

    # Process each pocket
    for id, voxels in pockets_dict.items():
        
        # Check each voxel in the pocket
        for voxel in voxels:
            i, j, k = voxel # Retrieve the indices of the voxel
            
            # Check if this voxel is a surface point
            is_surface = False
            
            for di, dj, dk in nearest_neighbors:
                neighbor = (i + di, j + dj, k + dk)
                
                # If neighbor is a protein voxel, this is a surface point
                if neighbor in voxel_grid and voxel_grid[neighbor] == -1:
                    is_surface = True
            
            # If this is a surface point, add it to our collection
            if is_surface:
                pockets_surface_dict[f'surface_{id}'].append(voxel)
                #surface_neighbors[voxel] = neighbor_coords

    print(pockets_surface_dict)
       
    return pockets_surface_dict

# DETERMINE ATOMS FOR EACH POCKET SURFACE
def find_atoms_for_pocket_surface(pockets_surface_dict, box, voxel_size=0.5):
    pockets_atoms_dict = {}

    xmin = box["X"][0]
    ymin = box["Y"][0]
    zmin = box["Z"][0]

    # Create a dictionary, keys: pockets, values: voxels in cartesian coordinates
    for id, voxels in pockets_surface_dict.items():
        coords = []
        for voxel in voxels:
            i, j, k = voxel # These are the voxel indices
            # Convert voxel index to Cartesian coordinate 
            x = xmin + i*voxel_size
            y = ymin + j*voxel_size
            z = zmin + k*voxel_size

            coords.append((x,y,z))
        
        # Store in dict
        # try as well:pocket_id.replace("surface_", "")
        pockets_atoms_dict[f'{id[7:]}'] = coords

    # Do we want to create a bounding box around each pocket? Now it should be doable since
    # we have the cartesian coordinates of the voxels and we can compute the dimensions of the box
    # How to make this box axis-aligned?
    # Once we have this box, prefilter atoms within this box
    # Use KDTree distance method to filter atoms
    # dictionary {'pocket_1: [(id, atom_name, (coorx, coordy, coordz), residue_id),
    # (id, atom_name, (coorx, coordy, coordz), residue_id), 
    # (id, atom_name, (coorx, coordy, coordz), residue_id),
    # (id, atom_name, (coorx, coordy, coordz), residue_id)]
    # 'pocket_2: [(id, atom_name, (coorx, coordy, coordz), residue_id),...],
    # 'pocket_3: [(id, atom_name, (coorx, coordy, coordz), residue_id),...]
    #}
    # Then loop in all tuples for each pockets, create a set called residues and add residue_id's to this set
    #{'pocket_1: [(residue_id1,residue_id2,residue_id3,... ),
    # 'pocket_2: [(residue_id1,...)],
    # 'pocket_3: [(residue_id2,...)]
    #}
    print(pockets_atoms_dict)
    return pockets_atoms_dict 


def visualize_pockets(pdb_file_path, pocket_surface, voxel_grid, atom_id_coords_dict, voxel_size=0.5, box=None, output_file=None):
    """
    Visualize protein structure and detected pockets in 3D.
    
    Parameters:
    -----------
    pdb_file_path : str
        Path to the PDB file (used for title)
    pockets : list
        List of pockets, where each pocket is a list of voxel coordinates
    voxel_grid : dict
        Dictionary mapping voxel coordinates to values
    atom_id_coords_dict : dict
        Dictionary mapping atom IDs to their 3D coordinates
    voxel_size : float
        Size of each voxel
    box : dict
        Bounding box dictionary with X, Y, Z ranges
    output_file : str, optional
        Path to save the visualization (if None, display instead)
    """
    # Convert back the pocket_surface dictionary to a list of lists to use visualization without further major changes
    pockets = []
    for pocket_surface in pocket_surface.values():
        pockets.append(pocket_surface)

    # Extract atom coordinates
    atom_coords = np.array(list(atom_id_coords_dict.values()))
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot protein atoms as small gray dots
    ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], 
               c='gray', s=1, alpha=0.3, label='Protein')
    
    # Define colors for different pockets
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Plot top pockets (up to 8)
    for i, pocket in enumerate(pockets[:min(len(pockets), len(colors))]):
        # Calculate pocket center
        pocket_voxels = np.array(pocket)
        
        # Convert voxel indices to Cartesian coordinates
        if box is not None:
            xmin, ymin, zmin = box["X"][0], box["Y"][0], box["Z"][0]
            pocket_coords = np.zeros((len(pocket_voxels), 3))
            for j, (x, y, z) in enumerate(pocket_voxels):
                # Convert voxel center to Cartesian coordinates
                pocket_coords[j, 0] = xmin + x * voxel_size + voxel_size / 2
                pocket_coords[j, 1] = ymin + y * voxel_size + voxel_size / 2
                pocket_coords[j, 2] = zmin + z * voxel_size + voxel_size / 2
        else:
            # If no box is provided, just use the voxel indices scaled by voxel_size
            pocket_coords = pocket_voxels * voxel_size
        
        # Plot pocket voxels
        ax.scatter(pocket_coords[:, 0], pocket_coords[:, 1], pocket_coords[:, 2],
                   c=colors[i], s=20, alpha=0.7, label=f'Pocket {i+1} ({len(pocket)} voxels)')
        
        # Calculate and plot pocket center
        center = np.mean(pocket_coords, axis=0)
        ax.scatter([center[0]], [center[1]], [center[2]], 
                   c=colors[i], s=100, edgecolor='black', marker='o')
        
        # Optional: Draw a sphere to represent the pocket volume
        # This is an approximation of the pocket shape
        pocket_size = len(pocket)
        r = (pocket_size * voxel_size**3 / (4/3 * np.pi))**(1/3)  # Approximate radius
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + r * np.cos(u) * np.sin(v)
        y = center[1] + r * np.sin(u) * np.sin(v)
        z = center[2] + r * np.cos(v)
        ax.plot_surface(x, y, z, color=colors[i], alpha=0.1)
    
    # Set labels and title
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Protein Structure and Detected Pockets for {pdb_file_path}')
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Set equal aspect ratio
    # Calculate the bounds
    max_range = np.array([
        atom_coords[:, 0].max() - atom_coords[:, 0].min(),
        atom_coords[:, 1].max() - atom_coords[:, 1].min(),
        atom_coords[:, 2].max() - atom_coords[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (atom_coords[:, 0].max() + atom_coords[:, 0].min()) / 2
    mid_y = (atom_coords[:, 1].max() + atom_coords[:, 1].min()) / 2
    mid_z = (atom_coords[:, 2].max() + atom_coords[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.tight_layout()
        plt.show()



# STEP 8
# Identify the aminoacids and atoms that surround the surface of a pocket


# STEP 9
# Prepare output and display
def calculate_pocket_properties(pocket, voxel_grid, bounding_box, voxel_size):
    """
    Calculate properties of a pocket such as volume, surface area, score, and centroid
    """
    xmin = bounding_box["X"][0]
    ymin = bounding_box["Y"][0]
    zmin = bounding_box["Z"][0]
    
    # Volume is simply the number of voxels times the voxel volume
    volume = len(pocket) * (voxel_size**3)
    
    # Calculate centroid
    sum_x, sum_y, sum_z = 0, 0, 0
    sum_score = 0
    
    for voxel in pocket:
        # Convert to Cartesian coordinates
        cart_x = xmin + voxel[0]*voxel_size + voxel_size/2
        cart_y = ymin + voxel[1]*voxel_size + voxel_size/2
        cart_z = zmin + voxel[2]*voxel_size + voxel_size/2
        
        sum_x += cart_x
        sum_y += cart_y
        sum_z += cart_z
        sum_score += voxel_grid[voxel]
    
    centroid = (sum_x/len(pocket), sum_y/len(pocket), sum_z/len(pocket))
    avg_score = sum_score/len(pocket)
    
    # Approximate surface area by counting surface voxels
    #surface_voxels, _ = determine_pocket_surface(pocket, voxel_grid, (float('inf'), float('inf'), float('inf')))
    #surface_area = len(surface_voxels) * (voxel_size**2)
    
    return {
        'volume': volume,
        #'surface_area': surface_area,
        'centroid': centroid,
        'score': avg_score
    }

# Generate PDB file for predicted pockets
def generate_pockets_pdb(pockets, voxel_grid, bounding_box, voxel_size, output_file):
    """
    Generate a PDB file representing the detected pockets
    """
    xmin = bounding_box["X"][0]
    ymin = bounding_box["Y"][0]
    zmin = bounding_box["Z"][0]
    
    with open(output_file, 'w') as f:
        f.write("REMARK  Generated by LIGSITE Binding Site Predictor\n")
        f.write("REMARK  Columns:\n")
        f.write("REMARK  HETATM  ID  Residue  Chain  ResNum  X      Y      Z      Occupancy  Score\n")
        
        atom_idx = 1
        for pocket_idx, pocket in enumerate(pockets):
            #properties = calculate_pocket_properties(pocket, voxel_grid, bounding_box, voxel_size)
            
            # Write REMARK for this pocket
            #f.write(f"REMARK  POCKET {pocket_idx+1} PROPERTIES: Volume={properties['volume']:.2f}, SurfaceArea={properties['surface_area']:.2f}\n")
            
            # Write voxels as atoms
            for voxel in pocket:
                # Convert to Cartesian coordinates
                x = xmin + voxel[0]*voxel_size + voxel_size/2
                y = ymin + voxel[1]*voxel_size + voxel_size/2
                z = zmin + voxel[2]*voxel_size + voxel_size/2
                
                # Score is the number of PSP events
                score = voxel_grid[voxel]
                
                # Write in PDB format
                f.write(f"HETATM{atom_idx:5d}  POC POC {chr(65+pocket_idx%26):1s}{pocket_idx+1:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{score:6.2f}\n")
                atom_idx += 1
            
            # Add TER record between pockets
            f.write(f"TER   {atom_idx:5d}      POC {chr(65+pocket_idx%26):1s}{pocket_idx+1:4d}\n")
            
# Generate PDB file for pocket residues
def generate_pocket_residues_pdb(pocket_idx, surrounding_residues, output_file):
    """
    Generate a PDB file containing the residues that form the pocket
    """
    with open(output_file, 'w') as f:
        f.write("REMARK  Residues forming the pocket\n")
        f.write("REMARK  CHAIN, RESNAME, RESNUM\n")

        # Write residue list in the header
        for (chain_id, res_id), res_info in surrounding_residues.items():
            f.write(f"REMARK  {chain_id} {res_info['name']} {res_id}\n")

        f.write("\n")

        # Write atom coordinates with strict PDB column alignment
        atom_idx = 1
        for (chain_id, res_id), res_info in surrounding_residues.items():
            for atom_id, atom_name, (x, y, z) in res_info['atoms']:
                # Format atom name according to PDB conventions (columns 13-16)
                # For standard formatting, right-justify element symbols
                if len(atom_name) == 4:
                    atom_name_formatted = atom_name
                else:
                    # Right-justify the element symbol (e.g., " CA " instead of "CA ")
                    atom_name_formatted = f" {atom_name.ljust(3)}"

                # Strict PDB format with exact column positioning
                # Columns: 1-6 (ATOM), 7-11 (atom serial), 13-16 (atom name), 18-20 (residue name),
                # 22 (chain ID), 23-26 (residue seq num), 31-38 (X), 39-46 (Y), 47-54 (Z),
                # 55-60 (occupancy), 61-66 (temperature factor)
                f.write(f"ATOM  {atom_idx:5d} {atom_name_formatted} {res_info['name']:3s} {chain_id:1s}{res_id:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}\n")
                atom_idx += 1
            
            # Add TER record after each residue chain
            f.write(f"TER   {atom_idx:5d}      {res_info['name']:3s} {chain_id:1s}{res_id:4d}\n")

# Generate PyMOL script
def generate_pymol_script(protein_file, pockets_file, output_file):
    """
    Generate a PyMOL script to visualize the protein and detected pockets
    """
    with open(output_file, 'w') as f:
        # Basic setup
        f.write("# PyMOL script for visualizing LIGSITE results\n")
        f.write(f"load {protein_file}, protein\n")
        f.write(f"load {pockets_file}, pockets\n")
        f.write("\n")
        
        # Style protein
        f.write("# Style protein\n")
        f.write("hide everything, protein\n")
        f.write("show cartoon, protein\n")
        f.write("color gray80, protein\n")
        f.write("\n")
        
        # Style pockets
        f.write("# Style pockets\n")
        f.write("hide everything, pockets\n")
        f.write("show spheres, pockets\n")
        
        # Different colors for each pocket
        colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", 
                 "wheat", "purple", "marine", "olive", "teal", "pink", "salmon"]
        
        for i, color in enumerate(colors):
            f.write(f"color {color}, pockets and chain {chr(65+i%26)}\n")
        
        f.write("\n")
        
        # Set transparency and sphere size
        f.write("# Set transparency and sphere size\n")
        f.write("set sphere_transparency, 0.5\n")
        f.write("set sphere_scale, 0.7\n")
        f.write("\n")
        
        # Center view
        f.write("# Center view\n")
        f.write("center\n")
        f.write("zoom\n")

# Complete workflow function
def run_complete_workflow(file_path, output_dir="./output", voxel_size=1.0, MIN_PSP=3, max_pockets=5):
    """
    Run the complete LIGSITE workflow for pocket detection
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing PDB file: {file_path}")
    print(f"Step 1: Getting atom coordinates and residue information...")
    atoms_ids_and_coordinates = atoms_coordinates_dict(file_path)

    # Get atoms and residues information
    atoms_coords, atom_id_coords_dict, residues = get_atoms_and_residues(file_path)
    
    print(f"Step 2: Creating grid and marking occupied voxels...")
    
    # Create bounding box and grid
    box, voxel_grid, grid_dimensions = create_bounding_box_and_voxels(atoms_ids_and_coordinates)

    # Mark occupied voxels (protein atoms)
    voxel_grid = mark_occupied_voxels(atoms_ids_and_coordinates, file_path)
    
    print(f"Step 3: Scanning along axes to detect protein-solvent-protein events...")
    
    # Scan along axes to detect PSP events
    axes = ['x','y','z']
    for axis in axes:
        voxel_grid = scan_along_axis(voxel_grid, grid_dimensions, axis)

    print(f"Step 4: Scanning along diagonals to detect protein-solvent-protein events...")
    
    # Scan along diagonals to detect PSP events
    diagonals = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)]
    for diagonal in diagonals:
        voxel_grid = scan_along_diagonal(grid_dimensions, voxel_grid, diagonal)
    
    #print(f"Step 5: Defining pockets (MIN_PSP = {MIN_PSP})...")
    
    # Define pockets
    pockets = define_pockets_and_cavities(voxel_grid, grid_dimensions)

    print(f"Found {len(pockets)} potential binding sites.")
    
    pockets = filter_pockets_by_size(pockets)
    print(f"Found {len(pockets)} filtered potential binding sites.")
    print(pockets)
    pocket_surface = determine_pocket_surface(voxel_grid, pockets)
    print(pocket_surface)

    find_atoms_for_pocket_surface(pocket_surface, box)

    # After detecting pockets
    visualize_pockets(file_path, pocket_surface, voxel_grid, atom_id_coords_dict, box=box)


    #print(f"Step 6: Determining pocket surfaces...")
    '''
    # Determine pocket surfaces
    pocket_surfaces = {}
    for i, pocket in enumerate(true_pockets):
        surface_voxels, connections = determine_pocket_surface(pocket, voxel_grid, grid_dimensions)
        pocket_surfaces[i] = (surface_voxels, connections)
    
    print(f"Step 7: Identifying surrounding residues...")
    
    # Identify surrounding residues
    pocket_residues = {}
    for i, pocket in enumerate(true_pockets):
        surface_voxels, _ = pocket_surfaces[i]
        surrounding_residues = identify_surrounding_residues(surface_voxels, bounding_box, voxel_size, residues)
        pocket_residues[i] = surrounding_residues
        
        print(f"  Pocket {i+1}: {len(surrounding_residues)} surrounding residues")
    '''
    #print(f"Step 8: Generating output files...")
    

    # Generate predicted pockets PDB file
    #pockets_pdb_file = os.path.join(output_dir, "predicted_pockets.pdb")
    #generate_pockets_pdb(pockets, voxel_grid, box, voxel_size, pockets_pdb_file)
    '''
    # Generate residue files for each pocket
    for i, surrounding_residues in pocket_residues.items():
        residues_pdb_file = os.path.join(output_dir, f"pocket_{i+1}_residues.pdb")
        generate_pocket_residues_pdb(i+1, surrounding_residues, residues_pdb_file)
    
    # Generate PyMOL script
    pymol_script_file = os.path.join(output_dir, "visualize.pml")
    generate_pymol_script(file_path, pockets_pdb_file, pymol_script_file)
    
    print(f"Workflow completed successfully!")
    print(f"Output files saved to: {output_dir}")
    print(f"To visualize in PyMOL, run: pymol -x {pymol_script_file}")
    '''
    #return pockets, cavities, voxel_grid, bounding_box
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
    else:
        print("Usage: python ligsite.py [pdb_file]")
        print("No PDB file specified, using default file '1a6u.pdb'")
        pdb_file = "1a6u.pdb"  # Default PDB file
    
    # Run the workflow
    run_complete_workflow(pdb_file)

