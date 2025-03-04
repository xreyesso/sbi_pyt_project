import sys
import math
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from collections import defaultdict

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

    
# def define_bounding_box(file_path, atom_coordinates):

#     """
#     Method to determine the coordinates of the box that will enclose the protein
#     """
#     x_coords = []
#     y_coords = []
#     z_coords = []

#     for atom_id, coordinates in atom_coordinates.items():
#             x_coords.append(coordinates[0])
#             y_coords.append(coordinates[1])
#             z_coords.append(coordinates[2])
    
#     xmin = min(x_coords)
#     xmax = max(x_coords)
#     ymin = min(y_coords)
#     ymax = max(y_coords)
#     zmin = min(z_coords)
#     zmax = max(z_coords)

#     bounding_box_dict = {"X":[xmin, xmax], "Y":[ymin, ymax], "Z":[zmin, zmax]}
#     return bounding_box_dict      

# The next step is to divide the 3D space (the bounding box in this case) into small cubes of 1 Angstrom per 
def create_bounding_box_and_voxels(atom_coordinates, voxel_size = 1.0):

    # box_coordinates = define_bounding_box(file_path, atom_coordinates)

    # xmin = box_coordinates["X"][0]
    # xmax = box_coordinates["X"][1]
    # ymin = box_coordinates["Y"][0]
    # ymax = box_coordinates["Y"][1]
    # zmin = box_coordinates["Z"][0]
    # zmax = box_coordinates["Z"][1]

    x_coords = []
    y_coords = []
    z_coords = []

    for _, coordinates in atom_coordinates.items():
            x_coords.append(coordinates[0])
            y_coords.append(coordinates[1])
            z_coords.append(coordinates[2])
    
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    zmin = min(z_coords)
    zmax = max(z_coords)

    bounding_box_dict = {"X":[xmin, xmax], "Y":[ymin, ymax], "Z":[zmin, zmax]}

    # Add 1 to ensure we cover the bounding box
    range_x = math.floor((xmax - xmin)/voxel_size) + 1
    range_y = math.floor((ymax - ymin)/voxel_size) + 1
    range_z = math.floor((zmax - zmin)/voxel_size) + 1
    print(f"Voxel grid dimensions are: rangex={range_x}, rangey={range_y}, rangez={range_z}")

    # Generate voxel coordinates and store them in a dictionary
    voxel_grid = {}
    for i in range(range_x): # recall: range(n) = 0, 1, 2, ..., n-1
        for j in range(range_y):
            for k in range(range_z):
                #Store in a dictionary with the indices i,j,k as key
                voxel_grid[(i,j,k)] = False
    
    return bounding_box_dict, voxel_grid

# The convex hull is the smallest convex polyhedron that encloses the protein
def create_convex_hull(atom_coordinates):
    atoms_list = []
    ids_list = [] # maps points in the convex hull to ids of the atoms (from the pdb)
    for id, coordinates in atom_coordinates.items():
        atoms_list.append((coordinates[0], coordinates[1], coordinates[2]))
        ids_list.append(id)
    atoms_np_array = np.array(atoms_list)
    hull = ConvexHull(atoms_np_array) # The ConvexHull method receives a numpy array as input
    return hull, ids_list

def identify_filled_empty_voxels(atom_coordinates, voxel_size = 1.0):
    """ 
    Step 4: Separate empty voxels from voxels filled by protein atoms in the convex hull
    """ 
    box, voxel_grid = create_bounding_box_and_voxels(atom_coordinates, voxel_size)

    # Map atoms to voxels
    atom_voxel_map = {}
    voxel_atom_map = defaultdict(list)
    
    xmin = box["X"][0]
    ymin = box["Y"][0]
    zmin = box["Z"][0]
    
    for identifier, coordinates in atom_coordinates.items():
        # Determine which voxel this atom is in
        i = math.floor((coordinates[0] - xmin) / voxel_size) # because indices start at 0
        j = math.floor((coordinates[1] - ymin) / voxel_size)
        k = math.floor((coordinates[2] - zmin) / voxel_size)
        
        voxel_idx = (i, j, k)
        voxel_grid[voxel_idx] = True
        atom_voxel_map[identifier] = voxel_idx
        voxel_atom_map[voxel_idx].append(identifier)
    
    # for id, atoms in voxel_atom_map.items():
    #     if len(atoms) > 1:
    #         print(voxel_atom_map[id])

    #print(voxel_atom_map)
    
    return voxel_grid, atom_voxel_map, voxel_atom_map
"""
def identify_nearest_empty_voxels(file_path, voxel_size=1.0, max_distance=5.0):
    """ """
    Identify empty voxels that are nearest to protein atoms within a maximum distance
    
    Parameters:
    - file_path: Path to the PDB file
    - voxel_size: Size of each voxel
    - max_distance: Maximum distance to consider an empty voxel as "near" to a protein atom
    
    Returns:
    - empty_voxels_near_protein: Dictionary with empty voxel indices as keys and distances to nearest protein atom as values
    """ """
    # Get voxel grid, atom-voxel mappings, and hull from existing function
    voxel_grid, atom_voxel_map, voxel_atom_map, hull = identify_filled_empty_voxels(file_path, voxel_size)
    
    # Initialize dictionary for empty voxels near protein
    empty_voxels_near_protein = {}
    
    # Get coordinates of all protein atoms
    atom_coords = {}
    for identifier, _, _, _, _, x, y, z in PDB_iterator(file_path):
        atom_coords[identifier] = (x, y, z)
    
    # Get box coordinates for reference
    box_coordinates = define_bounding_box(file_path)
    xmin, _ = box_coordinates["X"]
    ymin, _ = box_coordinates["Y"]
    zmin, _ = box_coordinates["Z"]
    
    # Identify empty voxels
    empty_voxels = [voxel_idx for voxel_idx, status in voxel_grid.items() if status == 'empty']
    
    # For each empty voxel, find distance to nearest protein atom
    for voxel_idx in empty_voxels:
        i, j, k = voxel_idx
        
        # Calculate center of voxel
        voxel_x = xmin + i*voxel_size + (voxel_size/2)
        voxel_y = ymin + j*voxel_size + (voxel_size/2)
        voxel_z = zmin + k*voxel_size + (voxel_size/2)
        
        # Find minimum distance to any protein atom
        min_distance = float('inf')
        
        for atom_id, (atom_x, atom_y, atom_z) in atom_coords.items():
            distance = math.sqrt((voxel_x - atom_x)*2 + (voxel_y - atom_y)2 + (voxel_z - atom_z)*2)
            min_distance = min(min_distance, distance)
        
        # If distance is within threshold, add to dictionary
        if min_distance <= max_distance:
            empty_voxels_near_protein[voxel_idx] = min_distance
    
    return empty_voxels_near_protein

"""
def define_pockets_from_triangles(hull, ids_list, atoms_coordinates, box, voxel_grid, voxel_atom_map, voxel_size=1.0):
    """
    Step 5: Define pockets by the volume generated by the vertices of each triangle on the convex hull
    
    Returns:
    - pockets: Dictionary with pocket IDs as keys and dictionaries with pocket information as values
    """

    xmin, xmax = box["X"]
    ymin, ymax = box["Y"]
    zmin, zmax = box["Z"]
    
    pockets = {}
    triangle_ids = hull.simplices
    
    for idx, triangle in enumerate(triangle_ids):
        # Get coordinates of triangle vertices
        p1 = atoms_coordinates[ids_list[triangle[0]]] # point 1 of the triangle
        p2 = atoms_coordinates[ids_list[triangle[1]]]
        p3 = atoms_coordinates[ids_list[triangle[2]]]
        
        # Define bounding box for triangle
        t_xmin = min(p1[0], p2[0], p3[0])
        t_xmax = max(p1[0], p2[0], p3[0])
        t_ymin = min(p1[1], p2[1], p3[1])
        t_ymax = max(p1[1], p2[1], p3[1])
        t_zmin = min(p1[2], p2[2], p3[2])
        t_zmax = max(p1[2], p2[2], p3[2])
        
        # Convert bounding box coordinates into voxel grid indices
        i_min = math.floor((t_xmin - xmin) / voxel_size)
        i_max = math.floor((t_xmax - xmin) / voxel_size) + 1
        j_min = math.floor((t_ymin - ymin) / voxel_size)
        j_max = math.floor((t_ymax - ymin) / voxel_size) + 1
        k_min = math.floor((t_zmin - zmin) / voxel_size)
        k_max = math.floor((t_zmax - zmin) / voxel_size) + 1
        
        # Find atoms and empty voxels within this pocket
        pocket_atoms = []
        pocket_empty_voxels = []
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    voxel_idx = (i, j, k)
                    if voxel_atom_map[voxel_idx]:
                        pocket_atoms.extend(voxel_atom_map[voxel_idx])
                    else:
                        pocket_empty_voxels.append(voxel_idx)
        
        # Calculate the normal vector of the triangle for depth calculation
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Store pocket information
        pockets[idx] = {
            'atoms': set(pocket_atoms),  # Remove duplicates
            'empty_voxels': set(pocket_empty_voxels),
            'triangle_vertices': [p1, p2, p3],
            'normal': normal,
            'boundary': {
                'x_range': (t_xmin, t_xmax),
                'y_range': (t_ymin, t_ymax),
                'z_range': (t_zmin, t_zmax)
            }
        }
    #print(pockets)
    return pockets

def compute_pocket_overlap(pockets, overlap_threshold=0.8):
    """
    Step 6: Compute overlap between pockets and merge those with high overlap
    
    Returns:
    - merged_pockets: Dictionary of merged pockets
    """
    # Computes overlap in both directions and stores it as a matrix
    overlap_matrix = {}
    for p1_id in pockets:
        overlap_matrix[p1_id] = {}
        for p2_id in pockets:
            if p1_id != p2_id:
                p1_atoms = pockets[p1_id]['atoms']
                p2_atoms = pockets[p2_id]['atoms']
                
                if len(p1_atoms) > 0:
                    overlap = len(p1_atoms.intersection(p2_atoms)) / len(p1_atoms)
                    overlap_matrix[p1_id][p2_id] = overlap
                else:
                    overlap_matrix[p1_id][p2_id] = 0.0
    
    # Merge pockets with overlap greater than threshold
    merged = {}  # Maps original pocket IDs to merged pocket IDs
    merged_pockets = {} # Dictionary from pocket id to pocket data
    next_merged_id = 0
    
    for p1_id in pockets:
        if p1_id in merged:
            continue
            
        # Create a new merged pocket
        merged_pocket = {
            'atoms': pockets[p1_id]['atoms'],
            #'empty_voxels': pockets[p1_id]['empty_voxels'],
            'constituent_pockets': [p1_id]
        }
        merged[p1_id] = next_merged_id

        pockets_merged_this_round = [p1_id]
        while pockets_merged_this_round:
            # check if the next_id pocket overlaps with any other pockets,
            # if it does, add to the merged pocket,
            # and add the overlapping pockets to the list of pockets merged this round.
            next_id = pockets_merged_this_round.pop()
        
            # Find all pockets that overlap with p1_id above threshold
            for p2_id in pockets:
                if p2_id not in merged:
                    if overlap_matrix[next_id][p2_id] >= overlap_threshold or overlap_matrix[p2_id][next_id] >= overlap_threshold:
                        merged[p2_id] = next_merged_id
                        merged_pocket['atoms'].update(pockets[p2_id]['atoms'])
                        #merged_pocket['empty_voxels'].update(pockets[p2_id]['empty_voxels'])
                        merged_pocket['constituent_pockets'].append(p2_id)
                        pockets_merged_this_round.append(p2_id)
                        print(f"merged {p2_id} overlapping {next_id} into {next_merged_id}: {merged_pocket['constituent_pockets']}")
        
        merged_pockets[next_merged_id] = merged_pocket
        next_merged_id += 1
    print(len(merged_pockets))
    return merged_pockets

def calculate_pocket_properties(merged_pockets, atoms_coordinates):
    """
    Step 7: Calculate physical properties of pockets (depth, surface area, volume)
    
    Returns:
    - pockets_with_properties: Dictionary of pockets with calculated properties
    """
    pockets_with_properties = {}
    
    for pocket_id, pocket in merged_pockets.items():
        # Calculate depth
        max_depth = 0
        pocket_atoms_coords = []
        
        for atom_id in pocket['atoms']:
            pocket_atoms_coords.append(atoms_coordinates[atom_id])
        
        # Calculate centroid of pocket atoms
        centroid = np.mean(pocket_atoms_coords, axis=0)
        
        # Calculate max distance from centroid to any atom (approximation of depth)
        for atom_coord in pocket_atoms_coords:
            distance = np.linalg.norm(np.array(atom_coord) - centroid)
            max_depth = max(max_depth, distance)
        
        # Approximate surface area (using convex hull of pocket atoms)
        if len(pocket_atoms_coords) >= 4:  # Need at least 4 points for 3D convex hull
            pocket_hull = ConvexHull(np.array(pocket_atoms_coords))
            surface_area = pocket_hull.area
            volume = pocket_hull.volume
        else:
            # Not enough points for convex hull
            continue
        
        # Store properties
        pockets_with_properties[pocket_id] = {
            **pocket, # Copy all data from pocket into this new dictionary
            'depth': max_depth,
            'surface_area': surface_area,
            'volume': volume
        }
    
    return pockets_with_properties

def identify_pocket_residues(pockets_with_properties, file_path):
    """
    Step 8: Identify residues corresponding to pocket atoms
    
    Returns:
    - pockets_with_residues: Dictionary of pockets with residue information
    """
    pockets_with_residues = {}
    
    # Create a mapping from atom ID to residue
    atom_to_residue = {}
    for identifier, _, residue_name, chain_ID, residue_ID, _, _, _ in PDB_iterator(file_path):
        residue_info = (residue_name, chain_ID, residue_ID)
        atom_to_residue[identifier] = residue_info
    
    for pocket_id, pocket in pockets_with_properties.items():
        residues = set()
        for atom_id in pocket['atoms']:
            residues.add(atom_to_residue[atom_id])
        
        # Store residue information
        pockets_with_residues[pocket_id] = {
            **pocket,
            'residues': residues,
        }
    
    return pockets_with_residues

def check_biochemical_conditions(pockets_with_residues, file_path):
    """
    Step 9: Check biochemical conditions for pockets
    
    Returns:
    - biochemically_filtered_pockets: Dictionary of pockets that meet biochemical criteria
    """
    # Define biochemical properties based on the paper's Table 1
    hydrogen_bond_acceptors = {
        'GLN': ['NE2'],
        'ASN': ['ND2'],
        'TYR': ['O']
    }
    
    hydrogen_bond_donors = {
        'THR': ['OG1'],
        'SER': ['OG'],
        'TYR': ['N', 'OH']
    }
    
    van_der_waals = {
        'MET': ['CE'],
        'ALA': ['CB'],
        'PRO': ['CB', 'CD', 'CG'],
        'LEU': ['CD1', 'CD2', 'CG'],
        'VAL': ['CG1', 'CG2', 'CB'],
        'ILE': ['CD1']
    }
    
    ionic = {
        'ASP': ['OD2'],
        'GLU': ['OE2'],
        'LYS': ['NZ'],
        'ARG': ['NH1', 'NH2']
    }
    
    sulfur = {
        'CYS': ['SG'],
        'MET': ['SD']
    }
    
    carbon_ring = {
        'HIS': ['CD1', 'CE1', 'CD2', 'CE2', 'CG'],
        'PHE': ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
        'TRP': ['CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3'],
        'TYR': ['CD1', 'CE1', 'CE2', 'CZ', 'CD2', 'CG']
    }
    
    # Initialize biochemical properties counters for each pocket
    biochemically_filtered_pockets = {}
    
    for pocket_id, pocket in pockets_with_residues.items():
        ha_bonds = 0  # Hydrogen acceptor bonds
        hd_bonds = 0  # Hydrogen donor bonds
        vdw_bonds = 0  # Van der Waals interactions
        ion_bonds = 0  # Ionic bonds
        sul_bonds = 0  # Sulfur bonds
        c_rings = 0   # Carbon rings
        
        # Count different types of potential interactions based on residues
        for residue_info in pocket['residues']:
            residue_name, chain_ID, residue_ID = residue_info
            
            # Check for hydrogen bond acceptors
            if residue_name in hydrogen_bond_acceptors:
                ha_bonds += 1
            
            # Check for hydrogen bond donors
            if residue_name in hydrogen_bond_donors:
                hd_bonds += 1
            
            # Check for van der Waals interactions
            if residue_name in van_der_waals:
                vdw_bonds += 1
            
            # Check for ionic interactions
            if residue_name in ionic:
                ion_bonds += 1
            
            # Check for sulfur interactions
            if residue_name in sulfur:
                sul_bonds += 1
            
            # Check for carbon rings
            if residue_name in carbon_ring:
                c_rings += 1
        
        # Store biochemical properties
        biochemical_properties = {
            'hydrogen_acceptor_bonds': ha_bonds,
            'hydrogen_donor_bonds': hd_bonds,
            'van_der_waals_bonds': vdw_bonds,
            'ionic_bonds': ion_bonds,
            'sulfur_bonds': sul_bonds,
            'carbon_rings': c_rings
        }
        
        # Determine if pocket is likely to be an active site based on biochemical properties
        # A simple criterion: must have at least one hydrogen bond potential and one other interaction
        is_potential_active_site = (ha_bonds + hd_bonds > 0) and (vdw_bonds + ion_bonds + sul_bonds + c_rings > 0)
        
        if is_potential_active_site:
            biochemically_filtered_pockets[pocket_id] = {
                **pocket,
                'biochemical_properties': biochemical_properties,
            }
    
    return biochemically_filtered_pockets

def run_pocket_detection(file_path, overlap_threshold=0.8):
    """
    Run the full pocket detection algorithm (steps 1-9)
    
    Returns:
    - final_pockets: Dictionary of filtered pockets that are likely to be active sites
    """
   
    atoms_ids_and_coordinates = atoms_coordinates_dict(file_path)

    # Create voxels
    box, voxel_grid = create_bounding_box_and_voxels(atoms_ids_and_coordinates)

    # Create convex hull
    hull, ids_list = create_convex_hull(atoms_ids_and_coordinates)

    # Identify filled and empty voxels
    voxel_grid, atom_voxel_map, voxel_atom_map = identify_filled_empty_voxels(atoms_ids_and_coordinates)

    # Define pockets from triangles
    pockets = define_pockets_from_triangles(hull, ids_list, atoms_ids_and_coordinates, box, voxel_grid, voxel_atom_map)
   
    # Compute pocket overlap and merge pockets
    merged_pockets = compute_pocket_overlap(pockets, overlap_threshold)

    # Calculate pocket properties
    pockets_with_properties = calculate_pocket_properties(merged_pockets, atoms_ids_and_coordinates)
    print(pockets_with_properties)
    print(len(pockets_with_properties))
    
    # Identify pocket residues
    pockets_with_residues = identify_pocket_residues(pockets_with_properties, file_path)
    
    # Check biochemical conditions
    final_pockets = check_biochemical_conditions(pockets_with_residues, file_path)
    
    return final_pockets

# Example usage:
file_path = "1a6u.pdb"
final_pockets = run_pocket_detection(file_path)
print(f"Found {len(final_pockets)} potential active site pockets")

#run_pocket_detection("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb")