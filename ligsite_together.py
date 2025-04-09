import numpy as np
import sys
import math
import os
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import datetime
import shutil


# OVERVIEW
# STEP 0: Read PDB file and get atoms and residues information.
# STEP 1: Create a bounding box for the protein and divide this space into a grid made of voxels.
# STEP 2: Set a value for each voxel. Start by setting all voxels to 0. A voxel that is inaccessible 
#         to solvent (because it is already occupied by the protein) gets a value of -1.
# STEP 3: A sequence of voxels which starts with protein, followed by solvent and ending with protein is 
#         called a protein-solvent-protein (PSP) event. First, scan along the x, y and z axis to detect PSP events.
# STEP 4: Scan along 4 cubic diagonals to detect more PSP events.
# STEP 5.1: Detect pockets as regions of voxels with a minimum number of PSP events (MIN_PSP) 
#           by checking nearest neighbors.
# STEP 5.2: Take out pockets that don't have the minimum number of voxels.
# STEP 6: Determine the voxels that are part of the surface of a pocket, 
#         by checking which voxels have protein neighbors.
# STEP 7: Identify the aminoacids and atoms that surround the surface of a pocket.
# STEP 8: Calculate the volume, surface area, depth and center for each pocket
# STEP 9: Prepare output and display. Print results summary on the terminal, generate a report file, individual pdb files per pocket
#         and a pymol script to visualize the results


van_der_Waals_radii = {
    "C": 1.70,
    "H": 1.17,
    "O": 1.40,
    "N": 1.50,
    "S": 1.85
}

color_list = [
        "red", "green", "blue", "yellow", "magenta", 
        "cyan", "orange", "hot pink", "lime green", "deep sky blue",
        "gold", "purple", "salmon", "forest green", "dodger blue",
        "olive", "sienna", "violet", "teal", "tomato",
        "navy blue", "chocolate", "spring green", "crimson", "medium purple"
    ]

# STEP 0
def PDB_iterator(pdb_file_path=None):
    """
    Read the input pdb file and extract information about the residues (id, name, chain id) 
    and atoms (id, name, coordinates).
    """
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
    """
    Create two dictionaries to store the atoms information and the residues information, respectively.
    """
    atom_id_coords_dict = {}
    residue_info_dict = {}
    
    for atom_id, atom_name, residue_name, chain_ID, residue_ID, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        # Store atom coordinates
        atom_id_coords_dict[atom_id] = (x_coordinate, y_coordinate, z_coordinate)
        
        # Determine the element type from atom name (first character usually)
        element = atom_name[0]
        
        # Create unique residue key (chain_ID, residue_ID)
        residue_key = (chain_ID, residue_ID)
        
        # Initialize residue entry if it does not exist
        if residue_key not in residue_info_dict:
            residue_info_dict[residue_key] = {
                'id': residue_ID,
                'name': residue_name,
                'chain': chain_ID,
                'atoms': [],
                'atom_coords': [],
                'elements': [],
                'atom_ids': []
            }
        
        # Add atom information to the residue
        residue_info_dict[residue_key]['atoms'].append(atom_name)
        residue_info_dict[residue_key]['atom_coords'].append((x_coordinate, y_coordinate, z_coordinate))
        residue_info_dict[residue_key]['elements'].append(element)
        residue_info_dict[residue_key]['atom_ids'].append(atom_id)
    
    return atom_id_coords_dict, residue_info_dict

# STEP 1    
def create_bounding_box_and_voxels(atom_coordinates, van_der_Waals_radii, voxel_size, probe_radius):
    """
    Create a bounding box for the protein and divide this 3D space into small cubes of x Angstrom per side.
    Each one of these small cubes is called a "voxel".
    """
    x_coords = []
    y_coords = []
    z_coords = []

    for _, coordinates in atom_coordinates.items():
            x_coords.append(coordinates[0])
            y_coords.append(coordinates[1])
            z_coords.append(coordinates[2])
   
    r_atom_max = max([radius for radius in van_der_Waals_radii.values()])

    r_max = r_atom_max + probe_radius

    xmin = min(x_coords) - r_max
    xmax = max(x_coords) + r_max
    ymin = min(y_coords) - r_max
    ymax = max(y_coords) + r_max
    zmin = min(z_coords) - r_max
    zmax = max(z_coords) + r_max

    bounding_box_dict = {"X":[xmin, xmax], "Y":[ymin, ymax], "Z":[zmin, zmax]}

    # Add 1 to ensure we cover the bounding box
    range_x = math.floor((xmax - xmin)/voxel_size) + 1
    range_y = math.floor((ymax - ymin)/voxel_size) + 1
    range_z = math.floor((zmax - zmin)/voxel_size) + 1

    # Generate voxel coordinates and store them in a dictionary
    voxel_grid = {}
    for i in range(range_x): # Recall: range(n) = 0, 1, 2, ..., n-1
        for j in range(range_y):
            for k in range(range_z):
                # Store voxels in a dictionary with the indices i,j,k as key
                voxel_grid[(i,j,k)] = 0
    
    return bounding_box_dict, voxel_grid, (range_x, range_y, range_z)

# STEP 2
def mark_occupied_voxels(residue_info_dict, box, voxel_grid, voxel_size, probe_radius):
    """
    A voxel is 'occupied' if there are protein atoms sufficiently close to it. Instead of checking all protein atoms, 
    we restrict the search to a bounding box for the atom that considers its van der Waals radius and the probe radius.
    """
    
    xmin = box["X"][0]
    ymin = box["Y"][0]
    zmin = box["Z"][0]
    
    # Iterate through each residue and its atoms
    for residue_key, residue_data in residue_info_dict.items():
        # Process each atom in the residue
        for idx, coords in enumerate(residue_data['atom_coords']):
            # Extract element from atom name (first character)
            element = residue_data['elements'][idx]
            
            # Get atom coordinates
            x, y, z = coords
            
            # Get van der Waals radius for this element
            r_atom = van_der_Waals_radii[element]
            r = r_atom + probe_radius

            # Convert atom's bounding box coordinates into voxel grid indices
            i0 = math.floor((x - r - xmin) / voxel_size)
            j0 = math.floor((y - r - ymin) / voxel_size)
            k0 = math.floor((z - r - zmin) / voxel_size)
            i1 = math.floor((x + r - xmin) / voxel_size) + 1
            j1 = math.floor((y + r - ymin) / voxel_size) + 1
            k1 = math.floor((z + r - zmin) / voxel_size) + 1
            
            # Iterate over the voxel range defined above
            for l in range(i0, i1 + 1):
                for m in range(j0, j1 + 1):
                    for n in range(k0, k1 + 1):
                        # Convert voxel index to Cartesian coordinate and add voxel_size/2 to each 
                        # coordinate to get the voxel center
                        voxel_x = xmin + l*voxel_size + voxel_size / 2
                        voxel_y = ymin + m*voxel_size + voxel_size / 2
                        voxel_z = zmin + n*voxel_size + voxel_size / 2

                        # Compute distance from voxel center to atom
                        distance = math.sqrt((voxel_x - x)**2 + (voxel_y - y)**2 + (voxel_z - z)**2)
                       
                        if distance <= r:
                            if (l, m, n) in voxel_grid:
                                # Mark voxel as -1, i.e. occupied
                                voxel_grid[(l, m, n)] = -1
    
    return voxel_grid

# STEP 3
def scan_along_axis(voxel_grid, grid_dimensions, axis):
    """
    A sequence of voxels which starts with protein, followed by solvent and ending with protein is 
    called a protein-solvent-protein (PSP) event.
    First, scan along the x, y and z axis to detect PSP events.
    All values in between enclosing -1's are increased by 1 in each scan.
    """

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
        for j in dim2_range: # For all lines parallel to the indicated axis
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
def scan_along_diagonal(grid_dimensions, voxel_grid, diagonal_vector):
    """
    Scan along four cubic diagonals to reduce the dependency on the protein's orientation.
    The scan direction is given by a diagonal_vector:
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

    return(voxel_grid)


# STEP 5.1
nearest_neighbors = [
        (1, 0, 0), (-1, 0, 0),  # x-axis neighbors
        (0, 1, 0), (0, -1, 0),  # y-axis neighbors
        (0, 0, 1), (0, 0, -1)   # z-axis neighbors
    ]

def cluster_into_pockets(voxel_grid, grid_dimensions, MIN_PSP):
    """
    Cluster the voxels into pockets by going through the voxels and their 
    neighbors, clustering the ones that are nearby and reach the minimum
    of protein-solvent-protein events.
    """
    
    visited = set()
    pockets = []
    range_x, range_y, range_z = grid_dimensions

    # Find all grid points with values >= MIN_PSP events (default is 3)
    for i in range(range_x):
        for j in range(range_y):
            for k in range(range_z):
                voxel = (i, j, k)
                
                # Skip if voxel coordinates have already been visited
                # or are below the MIN_PSP threshold
                if voxel in visited or voxel_grid.get(voxel, 0) < MIN_PSP:
                    continue
                
                # Start a new pocket with the initial voxel
                pocket = []
                queue = deque([voxel])
                visited.add(voxel)
                
                # Start a region growing process, where we will expand from the initial voxel
                # to a full pocket in all 6 dimensions
                while queue:
                    # Take the voxel that you want to explore from the queue,
                    # always in order of discovery
                    current = queue.popleft()
                    pocket.append(current)
                    
                    # Check the 6-connected neighbors of the voxel, and 
                    # whether they are in the grid and reach the MIN_PSP
                    for direction in nearest_neighbors:
                        di, dj, dk = direction
                        neighbor = (current[0]+di, current[1]+dj, current[2]+dk)
                        
                        if (neighbor not in visited and 
                            0 <= neighbor[0] < range_x and 
                            0 <= neighbor[1] < range_y and 
                            0 <= neighbor[2] < range_z and
                            voxel_grid.get(neighbor, 0) >= MIN_PSP):
                            
                            visited.add(neighbor)
                            queue.append(neighbor)  
                
                # Create a list of lists with the pockets
                if pocket:
                    pockets.append(pocket)
    
    # Sort pockets by size (largest first)
    return sorted(pockets, key=len, reverse=True)

# STEP 5.2
def filter_pockets_by_size(pockets, min_voxel_count=40):
    """
    Filter out pockets that have a voxel count lower than the minimum,
    due to them being too small
    """
    pockets_dict = {}
    n = 1
    # Iterate through the pockets
    for pocket in pockets:
        if len(pocket) >= min_voxel_count:
            # Save the final pockets into a dictionary
            pockets_dict[f'pocket_{n}'] = pocket
            n +=1
    return pockets_dict

# STEP 6
def determine_pockets_surface(voxel_grid, pockets_dict):
    """
    Check which voxels of each pocket are part of the surface of the pocket,
    by checking if any of their neighbors are protein voxels
    """
    pockets_surface_dict = defaultdict(list)

    # Iterate through the pockets
    for id, voxels in pockets_dict.items():
        
        # Iterate through the voxels of each pocket
        for voxel in voxels:
            i, j, k = voxel
            
            # Define is_surface as False
            is_surface = False
            
            # Check all the neighbors of the voxel to see if one of them is a protein
            for direction in nearest_neighbors:
                di, dj, dk = direction
                neighbor = (i + di, j + dj, k + dk)
                        
                # If neighbor is inside the grid and it is a protein,
                # the voxel we are looking at is a suface point
                if neighbor in voxel_grid and voxel_grid[neighbor] == -1:
                    is_surface = True
            
            # If the voxel is a surface point, add it to the pockets_surface_dict
            if is_surface:
                pockets_surface_dict[f'surface_{id}'].append(voxel)
   
    return pockets_surface_dict


# STEP 7
def identify_surrounding_residues_and_atoms(pocket_surfaces, residues, atom_id_coords_dict, bounding_box, voxel_size, probe_radius):
    """
    Identify the aminoacids and atoms that surround the surface of a pocket, by 
    finding the atoms which are at a certain distance of the surface voxels.
    """
    # Get the minimum coordinates of the bounding box
    xmin = bounding_box["X"][0]
    ymin = bounding_box["Y"][0]
    zmin = bounding_box["Z"][0]
    
    # Define the distance threshold for considering an atom as surrounding the pocket
    # This threshold is the probe-radius plus an average of the Van der Waals radii value
    distance_threshold = probe_radius + 1.7
    
    pocket_surroundings = {}
    
    # Iterate through each pocket surface
    for pocket_id, surface_points in pocket_surfaces.items():
        surrounding_residues = {}

        # Iterate through the voxels of each pocket surface
        for surface_voxel in surface_points:
            # Convert the voxels to cartesian coordinates (using the center of the voxel)
            i, j, k = surface_voxel
            voxel_x = xmin + i*voxel_size + voxel_size/2
            voxel_y = ymin + j*voxel_size + voxel_size/2
            voxel_z = zmin + k*voxel_size + voxel_size/2
            voxel_center = np.array([voxel_x, voxel_y, voxel_z])
        
            # Iteate through the atoms to find those close to this surface point
            for atom_id, coords in atom_id_coords_dict.items():
                atom_coords = np.array(coords)
                distance = np.linalg.norm(voxel_center - atom_coords)

                # If the atom is within the threshold distance, consider it as surrounding the pocket
                if distance <= distance_threshold:
                    
                    # Find which residue this atom belongs to
                    for residue_key, residue_info in residues.items():
                        for id in residue_info['atom_ids']:
                            if id == atom_id:
                                # Add this residue to our dictionary of surrounding residues for that pocket
                                if residue_key not in surrounding_residues:
                                    surrounding_residues[residue_key] = {
                                        'name': residue_info['name'],
                                        'atoms': []
                                    }
                                
                                # Add the atom that we know is close to the pocket to this residue (if it is not already there)
                                if atom_id not in [a[0] for a in surrounding_residues[residue_key]['atoms']]:
                                    atom_idx = residue_info['atom_ids'].index(atom_id)
                                    atom_name = residue_info['atoms'][atom_idx]
                                    surrounding_residues[residue_key]['atoms'].append((id,coords,atom_name))
                                break
                    
        # Sort by residue id
        surrounding_residues_sorted = dict(sorted(surrounding_residues.items(), key = lambda x: x[0][1]))

        # Sort atoms within each residue by atom ID
        for residue_key, residue_info in surrounding_residues_sorted.items():
            residue_info['atoms'] = sorted(residue_info['atoms'], key=lambda x: x[0])
        
        # Add the surrounding residues for a specific pocket
        pocket_surroundings[pocket_id] = surrounding_residues_sorted

    
    return pocket_surroundings

# STEP 8
def calculate_pocket_properties(pocket_voxels, pocket_surface, bounding_box, voxel_size):
    """
    Calculate pocket properties for a single pocket (volume, surface area, depth and center)
    """
    xmin = bounding_box["X"][0]
    ymin = bounding_box["Y"][0]
    zmin = bounding_box["Z"][0]
    
    distances = []

    # The volume of the pocket is equal to the number of voxels times the voxel volume
    volume = len(pocket_voxels) * (voxel_size**3)
    
    # Calculate center of the pocket
    sum_x, sum_y, sum_z = 0, 0, 0
    
    # Store the cartesian coordinates of each center of a surface voxel in a list
    coords = []
    for surface_voxel in pocket_surface:
        i, j, k = surface_voxel 
        # Convert voxel indices to cartesian coordinates (using the center of the voxel)
        x = xmin + i*voxel_size + voxel_size/2
        y = ymin + j*voxel_size + voxel_size/2
        z = zmin + k*voxel_size + voxel_size/2
        coords.append((x,y,z))

    for voxel in pocket_voxels:
        i, j, k = voxel # These are the voxel indices
        # Convert voxel tuple to Cartesian coordinates (coordinates of voxel center)
        cart_x = xmin + i * voxel_size + voxel_size / 2
        cart_y = ymin + j * voxel_size + voxel_size / 2
        cart_z = zmin + k * voxel_size + voxel_size / 2
        
        sum_x += cart_x
        sum_y += cart_y
        sum_z += cart_z

        # Get the minimum distance from voxel center to the centers of surface voxels
        distances.append(
            min([compute_distance_in_R3(cart_x, cart_y, cart_z, voxel_coords[0], voxel_coords[1], voxel_coords[2]) for voxel_coords in coords]))
    
    center = (sum_x/len(pocket_voxels), sum_y/len(pocket_voxels), sum_z/len(pocket_voxels))
    
    # Approximate surface area by counting surface voxels
    surface_area = len(pocket_surface) * (voxel_size**2)
    
    return {
        'volume': volume,
        'surface_area': surface_area,
        'depth': max(distances),
        'center': center,
    }


def calculate_all_pockets_info(pockets, pockets_surface, box, voxel_size):
    """
    Calculate pockets properties for all pockets
    """
    pockets_information = {}
    for i in range(1, len(pockets)+1):
        pocket_key = f'pocket_{i}' 
        surface_key = f'surface_pocket_{i}'
        pockets_information[pocket_key] = calculate_pocket_properties(
            pockets[pocket_key], pockets_surface[surface_key], box, voxel_size)
    return pockets_information

# STEP 9
def filter_pockets_by_number_of_residues(pockets_dict, pocket_surroundings, pocket_properties, min_residue_count=3):    
    filtered_pockets_dict = {}
    filtered_pocket_surroundings = {}
    filtered_pocket_properties = {}

    i=1
    # Iterate through each pocket
    for pocket_id, surrounding_residues in pocket_surroundings.items():
        # Extract the pocket ID without surface
        original_pocket_id = pocket_id.replace('surface_', '')

        # Check if the pocket has enough surrounding residues
        if len(surrounding_residues) >= min_residue_count:
            # Keep this pocket and its data
            filtered_pockets_dict[f"pocket_{i}"] = pockets_dict[original_pocket_id]
            filtered_pocket_surroundings[f"surface_pocket_{i}"] = surrounding_residues
            filtered_pocket_properties[f"pocket_{i}"] = pocket_properties[original_pocket_id]
            i += 1
    print(pocket_properties)
    print(filtered_pocket_properties)
    return filtered_pockets_dict, filtered_pocket_surroundings, filtered_pocket_properties


# STEP 10
def print_results_summary(properties, pocket_surroundings):
    """
    Generates a report summary which is printed in the terminal
    """
    print("=== Results Summary ===\n")
    
    for i in range(1, len(pocket_surroundings)+1):
        properties_key = f'pocket_{i}'
        pocket_surroundings_key = f'surface_pocket_{i}' 
        print(f"Pocket {i}")
        print(f"  Center: {properties[properties_key]['center']} Å")
        print(f"  Surface area: {properties[properties_key]['surface_area']} Å²")
        print(f"  Volume: {properties[properties_key]['volume']} Å³")
        print(f"  Depth:  {properties[properties_key]['depth']} Å")
        print(f"  Nearby residues: {len(pocket_surroundings[pocket_surroundings_key])}")
        residues = []
        for (_, res_id), surrounding_residues in pocket_surroundings[pocket_surroundings_key].items():
            resname = surrounding_residues['name']
            residues.append(resname + str(res_id))
        residues_string = ", ".join(residues)
        print(f"  Residues: {residues_string}")


def generate_pockets_report(output_file, voxel_size, probe_radius, van_der_Waals_dict, 
                            pockets_dict, pocket_surroundings, pocket_properties, min_psp=3, min_voxel_count=30):
    """
    Generates a full report which is stored in an output file
    """

    with open(output_file, 'w') as f:
        f.write("=============================================================\n")
        f.write("                   PREDICTED POCKETS REPORT                   \n")
        f.write("=============================================================\n\n")
        
        # Parameters section
        f.write("Parameters used for the prediction of pockets\n")
        f.write(f"Voxel Size: {voxel_size} Å\n")
        f.write(f"Probe Radius: {probe_radius} Å\n")
        f.write(f"Minimum protein-solvent-protein events (PSP): {min_psp}\n")
        f.write(f"Minimum voxel count per pocket: {min_voxel_count}\n\n")
        f.write(f"Van der Waals radii values per element:\n")

        for element, radius in van_der_Waals_dict.items():
            f.write(f"  {element}: {radius} Å\n")
        
        f.write(f"\nNumber of Detected Pockets: {len(pockets_dict)}\n\n")
        
        # Detailed information for each pocket
        for i in range(1, len(pocket_surroundings)+1):
            pocket_key = f'pocket_{i}'
            pocket_surroundings_key = f'surface_pocket_{i}'
            f.write(f"==== POCKET {i} ====\n\n")
            f.write(f"Number of voxels in the pocket: {len(pockets_dict[pocket_key])}\n\n")
            
            # Pocket properties
            f.write(f"Properties of the pocket:\n")
            center_x, center_y, center_z = pocket_properties[pocket_key]['center']
            f.write(f"  Center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f}) Å\n")
            f.write(f"  Surface area: {pocket_properties[pocket_key]['surface_area']:.2f} Å²\n")
            f.write(f"  Volume: {pocket_properties[pocket_key]['volume']:.2f} Å³\n")
            f.write(f"  Depth: {pocket_properties[pocket_key]['depth']:.2f} Å\n\n")
            
            # Surface of the pocket
            if pocket_surroundings_key in pocket_surroundings:
                
                # List of residues
                f.write(f"Residues in the surface of the pocket ({len(pocket_surroundings[pocket_surroundings_key])}):\n")
                residues_list = []
                for (chain_id, residue_id), surrounding_residues in pocket_surroundings[pocket_surroundings_key].items():
                    residue_name = surrounding_residues['name']
                    residues_list.append(f"{chain_id}-{residue_name}{residue_id}")
                
                if residues_list:
                    f.write(", ".join(residues_list) + "\n\n")
                else:
                    f.write("  No residues have been found in this pocket\n\n")
                
                # List of atoms in the surface of the pocket
                f.write(f"Atoms in the surface of the pocket:\n")
                atoms_text = []

                for (chain_id, residue_id), surrounding_residues in pocket_surroundings[pocket_surroundings_key].items():
                    atoms_text.append(f"    {surrounding_residues['name']}{residue_id}:\n")
                    for atom_id, coords, atom_name in list(surrounding_residues['atoms']):
                        atoms_text.append(f"        Atom {atom_id:>5}: {atom_name:>3} ({coords[0]:.4f},{coords[1]:.4f},{coords[2]:.4f})\n")
                    atoms_text.append("\n")

                if atoms_text:
                    f.write("".join(atoms_text))
                else:
                    f.write("  No atoms have been found in this pocket")
                
                f.write("\n\n")

            else:
                f.write("No surrounding residues or atoms information available.\n\n")
    
"""      
def generate_pocket_pdb(pocket_number, voxels, properties, bounding_box, voxel_size, output_file):
    
    Generate a PDB file for a single pocket with extra information.
    

    xmin = bounding_box["X"][0]
    ymin = bounding_box["Y"][0]
    zmin = bounding_box["Z"][0]
    
    # Prepare header information
    volume = properties.get('volume', 0)
    surface = properties.get('surface_area', 0)
    depth = properties.get('depth', 0)
    center = properties.get('center', (0,0,0))
    center_str = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"
    
    with open(output_file, 'w') as f:
        # Write header lines
        f.write(f"REMARK  Pocket {pocket_number} generated by Pocket Detector\n")
        f.write(f"REMARK  Properties: Volume={volume:.2f} Å³, Surface={surface:.2f} Å²\n")
        f.write(f"REMARK  Depth={depth:.2f} Å, Center={center_str} Å\n")
        f.write(f"REMARK  Total voxels: {len(voxels)}\n")
        
        # Write each voxel as a pseudo-atom
        for i, voxel in enumerate(voxels):
            # Convert voxel indices to cartesian coordinates
            x = xmin + voxel[0] * voxel_size + voxel_size / 2
            y = ymin + voxel[1] * voxel_size + voxel_size / 2
            z = zmin + voxel[2] * voxel_size + voxel_size / 2
            
            # Make score for color
            score = 1.0 - (i / len(voxels)) if len(voxels) > 1 else 1.0
            
            # Write voxel line
            f.write(f"HETATM{i+1:5d} VOXL PKT P{pocket_number:3d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 {score:5.2f}    "
                    f"V={volume:.2f} D={depth:.2f}\n")
        
        # End file
        f.write("END\n")
"""            

# MODIFY IT
def generate_pocket_residues_pdb(pocket_number, pocket_surroundings, output_file):
    """
    Generate a PDB file containing the residues surrounding a pocket.
    """
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"REMARK  Binding Pocket {pocket_number} - Surrounding Residues\n")
        f.write(f"REMARK  Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("REMARK  Format: CHAIN RESNAME RESNUM ATOMS\n")
        
        # Write list of residues in REMARK lines
        for (chain_id, res_id), surrounding_residues in pocket_surroundings.items():
            atom_count = len(surrounding_residues['atoms'])
            f.write(f"REMARK  {chain_id} {surrounding_residues['name']:3s} {res_id:4d} ({atom_count} atoms)\n")
        
        #ter_points = []
        
        # Group by chain and residue
        for (chain_id, res_id), surrounding_residues in pocket_surroundings.items():
            # Track where TER records should go
            #ter_points.append(atom_idx + len(res_info['atoms']))
            
            # Write all atoms 
            for atom_tuple in surrounding_residues['atoms']:
                atom_id = atom_tuple[0]
                x, y, z = atom_tuple[1]
                atom_name = atom_tuple[2]

                """
                atom_name = "UNK"
                if 'atoms' in res_info and idx < len(res_info['atoms']):
                    atom_name = res_info['atoms'][idx]
                elif 'elements' in res_info and idx < len(res_info['elements']):
                    atom_name = res_info['elements'][idx]
                """
                f.write(f"ATOM  {atom_id:5d} {str(atom_name):4s} {surrounding_residues['name']:3s} {chain_id}{res_id:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 {res_id/100:5.2f}      "
                        f"{chain_id}  \n")
        
        # Add TER lines between
        """for ter_point in ter_points[:-1]:  # All except the last one
            f.write(f"TER   {ter_point:5d}\n")"""
        f.write(f"END\n")

def generate_chimera_script_pockets(protein_pdb, pocket_files, output_script):
    """
    Generate a Chimera script to visualize the original protein structure along with all pocket residues.
    """
    with open(output_script, 'w') as f:
        # Write header and comments
        f.write("# Chimera script for visualizing Protein Binding Pockets\n")
        f.write(f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# To use: open Chimera and run 'open <this_file>'\n\n")
        
        # Open the original protein PDB file
        f.write(f"open {protein_pdb}\n")
        
        # Set up the visualization style for the protein - now with surface
        f.write("\n# Set up the visualization style for the protein\n")
        f.write("color skyblue #0\n")  # Color protein skyblue
        f.write("surface #0\n")  # Add surface to the protein
        
        # Create an expanded color list for up to 25 pockets
        color_list = [
            "red", "green", "blue", "yellow", "magenta", 
            "cyan", "orange", "hot pink", "lime green", "deep sky blue",
            "gold", "purple", "salmon", "forest green", "dodger blue",
            "olive", "sienna", "violet", "teal", "tomato",
            "navy blue", "chocolate", "spring green", "crimson", "medium purple"
        ]
        
        # Load and visualize each pocket
        f.write("\n# Load and visualize pocket surfaces\n")
        for i, pocket_file in enumerate(pocket_files):
            model_id = i + 1
            pocket_color = color_list[i % len(color_list)]
            f.write(f"open {pocket_file}\n")
            f.write(f"~bond #{model_id}\n")
            f.write(f"represent sphere #{model_id} \n")
            f.write(f"color {pocket_color} #{model_id}\n")
            f.write(f"transparency 70,s #{model_id}\n")
            f.write(f"setattr r 1.0 #{model_id}\n")
        
        # Create labels for pockets
        f.write("\n# Create labels for pockets\n")
        for i, pocket_file in enumerate(pocket_files):
            pocket_num = i + 1
            pocket_color = color_list[i % len(color_list)]
            f.write(f"2dlabels create pocket{pocket_num} text \"Pocket {pocket_num}\" xpos 0.1 ypos {0.9 - 0.05*i} color {pocket_color}\n")
        
        # Set up view parameters
        f.write("\n# Set up view parameters\n")
        f.write("set bg_color white\n")
        f.write("set silhouette\n")  # Add silhouette edges
        f.write("set subdivision 1\n")  # Smooth molecular surface
        
        # Center and zoom on the protein
        f.write("\n# Center on the protein\n")
        f.write("focus\n")
        f.write("turn y 30\n")
        f.write("turn x 20\n") 

def start_chimera_script(protein_pdb, f):
    """
    Generate a Chimera script to visualize the original protein structure along with all pocket residues.
    """
    # Write header and comments
    f.write("# Chimera script for visualizing Protein Binding Pockets\n")
    f.write(f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("# To use: open Chimera and run 'open <this_file>'\n\n")
    
    # Open the original protein PDB file
    f.write(f"open {protein_pdb}\n")
    
    # Set up the visualization style for the protein
    f.write("\n# Set up ball & stick visualization style for the protein\n")
    f.write("color skyblue #0\n")  # Color protein skyblue
    f.write("~ribbon #0\n")  # Hide ribbon representation
    f.write("~surface #0\n")  # Hide any surface representation
    f.write("show #0\n")  # Show protein atoms
    f.write("represent bs #0\n")  # Represent as ball & stick
    f.write("~display solvent\n")


    # Load and visualize pocket residues
    f.write("\n# Load and visualize pocket residues\n")
    
         
def end_chimera_script(pocket_surroundings, color_list, f):
    # Create labels for pockets with max 16 pockets per column
    f.write("\n# Create labels for pockets with max 16 per column\n")
    num_pockets = len(pocket_surroundings)
    max_pockets_per_column = 16
    
    for i, _ in enumerate(pocket_surroundings):
        pocket_num = i + 1
        color = color_list[i % len(color_list)]
        
        # Calculate column number (0-based) and position within column
        column = i // max_pockets_per_column
        position_in_column = i % max_pockets_per_column
        
        # Only create a new column if we exceed 16 pockets
        if num_pockets <= max_pockets_per_column:
            # Just one column if 16 or fewer pockets
            xpos = 0.1
            ypos = 0.9 - 0.05 * i
        else:
            # Multiple columns if more than 16 pockets
            xpos = 0.1 + 0.4 * column
            ypos = 0.9 - 0.05 * position_in_column
            
        f.write(f"2dlabels create pocket{pocket_num} text \"Pocket {pocket_num}\" xpos {xpos} ypos {ypos} color {color}\n")
    # Set up view parameters
    f.write("\n# Set up view parameters\n")
    
    # Add lighting effects
    f.write("\n# Enhance lighting\n")
    f.write("light mode two-point\n")
    
    # Center and zoom on the protein
    f.write("\n# Center on the protein\n")
    f.write("focus\n")
    f.write("turn y 30\n")
    f.write("turn x 20\n")
        
def chimera_script_residues_bs(pocket_surroundings, color_list, f):
    """
    Generate a Chimera script to visualize the original protein structure along with all pocket residues.
    """
    # Load and visualize pocket residues
    residues = []
    for i in range(1, len(pocket_surroundings)+1):
        pocket_surroundings_key = f'surface_pocket_{i}'
        color = color_list[i % len(color_list)-1]
        initial_string = f"color {color} "
        final_string = ""

        for (chain_id, res_id) in pocket_surroundings[pocket_surroundings_key].keys():
            residues.append(f":{res_id}.{chain_id}")
        residues_string = " ".join(residues)
        final_string = initial_string + residues_string + "\n"
        f.write(final_string)

        initial_string = ""
        final_string = ""
        residues_string = ""
        residues = []   

 
def chimera_script_atoms_bs(pocket_surroundings, color_list, f):
    # Load and visualize pocket residues
    residues = []
    for i in range(1, len(pocket_surroundings)+1):
        residues_info_key = f'surface_pocket_{i}'
        color = color_list[i % len(color_list)-1]
        initial_string = f"color {color} "
        final_string = ""

        for (chain_id, res_id), surrounding_residues in pocket_surroundings[residues_info_key].items():

            for atom_tuple in surrounding_residues['atoms']:
                atom_name = atom_tuple[2]
                residues.append(f":{res_id}.{chain_id}@{atom_name}")

        residues_string = " ".join(residues)
        final_string = initial_string + residues_string + "\n"
        f.write(final_string)

        initial_string = ""
        final_string = ""
        residues_string = ""
        residues = []   




def generate_chimera_script_atoms_bs(protein_pdb, pocket_surroundings, color_list, output_script):
    with open(output_script, 'w') as f:
        start_chimera_script(protein_pdb, f)
        chimera_script_atoms_bs(pocket_surroundings, color_list, f)
        end_chimera_script(pocket_surroundings, color_list, f)

def generate_chimera_script_residues_bs(protein_pdb, pocket_surroundings, color_list, output_script):
    with open(output_script, 'w') as f:
        start_chimera_script(protein_pdb, f)
        chimera_script_residues_bs(pocket_surroundings, color_list, f)
        end_chimera_script(pocket_surroundings, color_list, f)

def generate_chimera_script_atoms_surface(protein_pdb, pocket_surroundings, color_list, output_script):
    with open(output_script, 'w') as f:
        start_chimera_script(protein_pdb, f)
        chimera_script_atoms_bs(pocket_surroundings, color_list, f)
        f.write("surface")
        f.write("~repr bs")
        end_chimera_script(pocket_surroundings, color_list, f)

def generate_chimera_script_residues_surface(protein_pdb, pocket_surroundings, color_list, output_script):
    with open(output_script, 'w') as f:
        start_chimera_script(protein_pdb, f)
        chimera_script_residues_bs(pocket_surroundings, color_list, f)
        f.write("surface")
        f.write("~repr bs")
        end_chimera_script(pocket_surroundings, color_list, f)



def visualize_pockets(pdb_file_path, pocket_surface, atom_id_coords_dict, voxel_size, box):

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
    colors = [
            "red", "green", "blue", "yellow", "magenta", 
            "cyan", "orange", "pink"]
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
    
    
    plt.tight_layout()
    plt.show()

def compute_distance_in_R3(x_1, y_1, z_1, x_2, y_2, z_2):
    return math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2 + (z_2 - z_1)**2)



def run_complete_workflow(file_path, output_dir="./output", voxel_size=0.5, MIN_PSP=3, probe_radius = 1.4):
    """
    Complete workflow to find the pockets of the protein
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the original PDB file to the output directory
    pdb_filename = os.path.basename(file_path)
    output_pdb_path = os.path.join(output_dir, pdb_filename)
    shutil.copy2(file_path, output_pdb_path)
    print(f"Copied original PDB file to: {output_pdb_path}")


    print(f"Processing PDB file: {file_path}")
    print(f"Step 1: Getting atom coordinates and residue information...")
    atoms_ids_and_coordinates, residues_info_dict = atoms_coordinates_dict(file_path)
    
    print(f"Step 2: Creating grid and marking occupied voxels...")
    # Create bounding box and grid
    box, voxel_grid, grid_dimensions = create_bounding_box_and_voxels(atoms_ids_and_coordinates, van_der_Waals_radii, voxel_size, probe_radius)

    # Mark occupied voxels (protein atoms)
    voxel_grid = mark_occupied_voxels(residues_info_dict, box, voxel_grid, voxel_size, probe_radius)
    
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
    
    # Find potential pockets
    print(f"Step 5: Detecting potential pockets...")

    pockets_list = cluster_into_pockets(voxel_grid, grid_dimensions, MIN_PSP)
    print(f"        Found {len(pockets_list)} regions of grid points with a minimum of {MIN_PSP} psp events.")

    pockets_dict = filter_pockets_by_size(pockets_list,)
    print(f"        Found {len(pockets_dict)} pockets.")
    
    # Determine the pocket surface
    print(f"Step 6: Extracting pocket surface voxels...")
    pockets_surface = determine_pockets_surface(voxel_grid, pockets_dict)

    # Determine the surrounding residues and atoms in each pocket
    print(f"Step 7: Extracting the surrounding residues and atoms per pocket...")
    pockets_residues_info_dict = identify_surrounding_residues_and_atoms(pockets_surface, residues_info_dict, atoms_ids_and_coordinates, 
                                                                         box, voxel_size, probe_radius)
    
    # Compute the properties (volume, surface area, depth, center) for all pockets
    print(f"Step 8: Computing the properties (volume, surface area, depth, center) for all pockets...\n")
    pockets_properties_info_dict = calculate_all_pockets_info(pockets_dict, pockets_surface, box, voxel_size)
    
    print(f"Step 9: Filtering pockets by number of residues")
    pockets_dict, pockets_residues_info_dict, pockets_properties_info_dict =  filter_pockets_by_number_of_residues(pockets_dict, pockets_residues_info_dict, pockets_properties_info_dict)    
    print("Found pockets")
    
    # Generate output files
    print(f"\nStep 10: Generating output files...")
    
    # Print a results summary in stdout
    print_results_summary(pockets_properties_info_dict, pockets_residues_info_dict)

    report_file = os.path.join(output_dir, "pockets_report.txt")
    generate_pockets_report(report_file, voxel_size, probe_radius, van_der_Waals_radii, pockets_dict,
                            pockets_residues_info_dict, pockets_properties_info_dict, MIN_PSP, min_voxel_count=30)

    # Generate individual PDB files for each pocket and its residues
    pocket_files = []
    residue_files = []
    
    for pocket_surface_id, _ in pockets_residues_info_dict.items():
        pocket_num = int(pocket_surface_id.split('_')[2])
        pocket_id = f"pocket_{pocket_num}"
        properties = pockets_properties_info_dict[pocket_id]
        
        """# Generate pocket PDB file
        pocket_file = os.path.join(output_dir, f"pocket_{pocket_num}.pdb")
        generate_pocket_pdb(pocket_num, voxels, properties, box, voxel_size, pocket_file)
        pocket_files.append(f"pocket_{pocket_num}.pdb")
        """
        # Generate residues PDB file
        residues = pockets_residues_info_dict.get(pocket_surface_id, {})
        residue_file = os.path.join(output_dir, f"pocket_{pocket_num}_residues.pdb")
        generate_pocket_residues_pdb(pocket_num, residues, residue_file)
        residue_files.append(f"pocket_{pocket_num}_residues.pdb") 

    # Generate comprehensive Chimera script
    chimera_script_pockets = os.path.join(output_dir, "pockets_visualization.cmd")
    chimera_script_residues_bs = os.path.join(output_dir, "bs_residues_pockets_visualization.cmd")
    chimera_script_atoms_bs = os.path.join(output_dir, "bs_atoms_pockets_visualization.cmd")
    chimera_script_residues_surface = os.path.join(output_dir, "surface_residues_pockets_visualization.cmd")
    chimera_script_atoms_surface = os.path.join(output_dir, "surface_atoms_pockets_visualization.cmd")
    
    generate_chimera_script_pockets(file_path, pocket_files, chimera_script_pockets)

    generate_chimera_script_residues_bs(file_path, pockets_residues_info_dict, color_list, chimera_script_residues_bs)
    generate_chimera_script_atoms_bs(file_path, pockets_residues_info_dict, color_list, chimera_script_atoms_bs)
    generate_chimera_script_residues_surface(file_path, pockets_residues_info_dict, color_list, chimera_script_residues_surface)
    generate_chimera_script_atoms_surface(file_path, pockets_residues_info_dict, color_list, chimera_script_atoms_surface)

    print(f"\nResults saved to {output_dir}")
    

    # Visualize pockets
    visualize_pockets(file_path, pockets_surface, atoms_ids_and_coordinates, voxel_size, box)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
        protein = pdb_file.split('.')[0]
        output_dir = f"./{protein}_output"
        # Run the workflow
        run_complete_workflow(pdb_file, output_dir)

    else:
        print("Usage: python find_pockets.py [pdb_file]")
        
    
    

