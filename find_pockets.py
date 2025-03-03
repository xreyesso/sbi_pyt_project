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

def create_convex_hull(file_path):
    atoms_list = []
    for _ , _, _, _, _, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        atoms_list.append((x_coordinate,y_coordinate,z_coordinate))
    atoms_np_array = np.array(atoms_list)
    hull = ConvexHull(atoms_np_array) # The ConvexHull method receives a numpy array as input
    return hull

def plot_convex_hull(file_path):
    atoms_list = []
    for _ , _, _, _, _, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        atoms_list.append((x_coordinate,y_coordinate,z_coordinate))
    atoms_np_array = np.array(atoms_list)
    hull = ConvexHull(atoms_np_array) # The ConvexHull method receives a numpy array as input
    #return hull

    # Plot the convex hull
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot_trisurf(
        np.array([atom[0] for atom in atoms_list]),
        np.array([atom[1] for atom in atoms_list]),
        np.array([atom[2] for atom in atoms_list]),
        triangles = np.array(hull.simplices),
        linewidth = 0.2, antialiased=True)

    plt.show()


#print(create_convex_hull("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb"))

def plot_convex_hull_with_vertices(file_path):
    #atoms_list = get_atom_coordinates(file_path)

    atoms_list = []
    for _ , _, _, _, _, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        atoms_list.append((x_coordinate,y_coordinate,z_coordinate))
    
    atoms_np_array = np.array(atoms_list)
    hull = ConvexHull(atoms_np_array)  # Compute the convex hull

    # Extract hull vertices
    hull_vertices_coordinates = np.array([hull.points[vertex] for vertex in hull.vertices])

    # Plot the convex hull
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_trisurf(
        np.array([atom[0] for atom in atoms_list]),
        np.array([atom[1] for atom in atoms_list]),
        np.array([atom[2] for atom in atoms_list]),
        triangles=hull.simplices,
        linewidth=0.2, antialiased=True, alpha=0.5
    )

    # Scatter plot the convex hull vertices
    ax.scatter(
        hull_vertices_coordinates[:, 0], 
        hull_vertices_coordinates[:, 1], 
        hull_vertices_coordinates[:, 2], 
        color='red', marker='o', s=50, label="Hull Vertices"
    )

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


def get_atom_coordinates(file_path):
    atoms_list = []
    ids_list = []
    
    # Iterate through PDB file and extract atom coordinates
    for identifier, _, _, _, _, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        atoms_list.append((x_coordinate, y_coordinate, z_coordinate))
        ids_list.append(identifier)
    return atoms_list, ids_list
    

## FUNCTION calls for testing/debugging
#print(create_convex_hull("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb"))
plot_convex_hull("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb")
#plot_convex_hull_with_vertices("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb")

file_path = "/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb"
hull = create_convex_hull(file_path)
triangle_ids = np.array(hull.simplices)
atom_list, ids_list = get_atom_coordinates(file_path)
for triangle in triangle_ids:
    print((ids_list[triangle[0]], ids_list[triangle[1]], ids_list[triangle[2]]), atom_list[triangle[0]], atom_list[triangle[1]], atom_list[triangle[2]])

# The hull vertices can be found using vertices attribute(return indices for input points):
hull_vertices_coordinates = [hull.points[vertex] for vertex in hull.vertices]
for point in hull_vertices_coordinates:
    print(point[0])

                      
# For each triangle, define a box around it→ axis aligned bounding box (aabb)
# get the vertices of the convex hull from the result returned from the scipy.spatial.ConvexHull function

def get_bounding_box_for_triangle(x_0, y_0, z_0, x_1, y_1, z_1, x_2, y_2, z_2):
    xmin = min(x_0, x_1, x_2)
    xmax = max(x_0, x_1, x_2)
    ymin = min(y_0, y_1, y_2)
    ymax = max(y_0, y_1, y_2)
    zmin = min(z_0, z_1, z_2)
    zmax = max(z_0, z_1, z_2)

    bounding_box_dict = {"X":[xmin, xmax], "Y":[ymin, ymax], "Z":[zmin, zmax]}
    return bounding_box_dict      




# Functions already implemented in your code:
# - PDB_iterator
# - define_bounding_box
# - create_voxels
# - create_convex_hull
# - get_atom_coordinates

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
    
    for idx, (identifier, _, _, _, _, x, y, z) in enumerate(PDB_iterator(file_path)):
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

def define_pockets_from_triangles(hull, atoms_list, file_path, voxel_size=1.0):
    """
    Step 5: Define pockets by the volume generated by the vertices of each triangle on the convex hull
    
    Returns:
    - pockets: Dictionary with pocket IDs as keys and dictionaries with pocket information as values
    """
    box_coordinates = define_bounding_box(file_path)
    xmin, xmax = box_coordinates["X"]
    ymin, ymax = box_coordinates["Y"]
    zmin, zmax = box_coordinates["Z"]
    
    pockets = {}
    triangle_ids = np.array(hull.simplices)
    
    for idx, triangle in enumerate(triangle_ids):
        # Get coordinates of triangle vertices
        p1 = atoms_list[triangle[0]]
        p2 = atoms_list[triangle[1]]
        p3 = atoms_list[triangle[2]]
        
        # Define bounding box for triangle
        t_xmin = min(p1[0], p2[0], p3[0])
        t_xmax = max(p1[0], p2[0], p3[0])
        t_ymin = min(p1[1], p2[1], p3[1])
        t_ymax = max(p1[1], p2[1], p3[1])
        t_zmin = min(p1[2], p2[2], p3[2])
        t_zmax = max(p1[2], p2[2], p3[2])
        
        # Calculate voxel coordinates for the pocket
        i_min = max(0, int((t_xmin - xmin) / voxel_size))
        i_max = min(int((t_xmax - xmin) / voxel_size) + 1, math.ceil((xmax - xmin)/voxel_size))
        j_min = max(0, int((t_ymin - ymin) / voxel_size))
        j_max = min(int((t_ymax - ymin) / voxel_size) + 1, math.ceil((ymax - ymin)/voxel_size))
        k_min = max(0, int((t_zmin - zmin) / voxel_size))
        k_max = min(int((t_zmax - zmin) / voxel_size) + 1, math.ceil((zmax - zmin)/voxel_size))
        
        # Find atoms and empty voxels within this pocket
        pocket_atoms = []
        pocket_empty_voxels = []
        
        voxel_grid, _, voxel_atom_map, _ = identify_filled_empty_voxels(file_path)
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    voxel_idx = (i, j, k)
                    if voxel_idx in voxel_grid:
                        # Check if this voxel is inside the triangle's bounding box
                        if voxel_grid[voxel_idx] == 'filled':
                            if voxel_idx in voxel_atom_map:
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
            'atoms': list(set(pocket_atoms)),  # Remove duplicates
            'empty_voxels': pocket_empty_voxels,
            'triangle_vertices': [p1, p2, p3],
            'normal': normal,
            'boundary': {
                'x_range': (t_xmin, t_xmax),
                'y_range': (t_ymin, t_ymax),
                'z_range': (t_zmin, t_zmax)
            }
        }
    
    return pockets

def compute_pocket_overlap(pockets, overlap_threshold=0.8):
    """
    Step 6: Compute overlap between pockets and merge those with high overlap
    
    Returns:
    - merged_pockets: Dictionary of merged pockets
    """
    overlap_matrix = {}
    for p1_id in pockets:
        overlap_matrix[p1_id] = {}
        for p2_id in pockets:
            if p1_id != p2_id:
                p1_atoms = set(pockets[p1_id]['atoms'])
                p2_atoms = set(pockets[p2_id]['atoms'])
                
                if len(p1_atoms) > 0:
                    overlap = len(p1_atoms.intersection(p2_atoms)) / len(p1_atoms)
                    overlap_matrix[p1_id][p2_id] = overlap
                else:
                    overlap_matrix[p1_id][p2_id] = 0.0
    
    # Merge pockets with overlap greater than threshold
    merged = {}  # Maps original pocket IDs to merged pocket IDs
    merged_pockets = {}
    next_merged_id = 0
    
    for p1_id in pockets:
        if p1_id in merged:
            continue
            
        # Create a new merged pocket
        merged_pocket = {
            'atoms': set(pockets[p1_id]['atoms']),
            'empty_voxels': list(pockets[p1_id]['empty_voxels']),
            'constituent_pockets': [p1_id]
        }
        merged[p1_id] = next_merged_id
        
        # Find all pockets that overlap with p1_id above threshold
        for p2_id in pockets:
            if p1_id != p2_id and p2_id not in merged:
                if p1_id in overlap_matrix and p2_id in overlap_matrix[p1_id]:
                    overlap = overlap_matrix[p1_id][p2_id]
                    if overlap >= overlap_threshold:
                        merged[p2_id] = next_merged_id
                        merged_pocket['atoms'].update(pockets[p2_id]['atoms'])
                        merged_pocket['empty_voxels'].extend(pockets[p2_id]['empty_voxels'])
                        merged_pocket['constituent_pockets'].append(p2_id)
        
        # Convert atoms back to list
        merged_pocket['atoms'] = list(merged_pocket['atoms'])
        merged_pockets[next_merged_id] = merged_pocket
        next_merged_id += 1
    
    return merged_pockets

def calculate_pocket_properties(merged_pockets, file_path):
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
        
        # Get coordinates of atoms in pocket
        atom_coords_dict = {}
        for _, atom_name, _, _, _, x, y, z in PDB_iterator(file_path):
            atom_coords_dict[atom_name] = (x, y, z)
        
        for atom_id in pocket['atoms']:
            if atom_id in atom_coords_dict:
                pocket_atoms_coords.append(atom_coords_dict[atom_id])
        
        # If there are no atoms in pocket, skip
        if not pocket_atoms_coords:
            continue
        
        # Calculate centroid of pocket atoms
        centroid = np.mean(pocket_atoms_coords, axis=0)
        
        # Calculate max distance from centroid to any atom (approximation of depth)
        for atom_coord in pocket_atoms_coords:
            distance = np.linalg.norm(np.array(atom_coord) - centroid)
            max_depth = max(max_depth, distance)
        
        # Approximate surface area (using convex hull of pocket atoms)
        if len(pocket_atoms_coords) >= 4:  # Need at least 4 points for 3D convex hull
            try:
                pocket_hull = ConvexHull(np.array(pocket_atoms_coords))
                surface_area = pocket_hull.area
                volume = pocket_hull.volume
            except:
                # If convex hull calculation fails, use simpler approximations
                surface_area = len(pocket['atoms']) * 10  # Rough estimate
                volume = len(pocket['atoms']) * 8  # Rough estimate
        else:
            # Not enough points for convex hull
            surface_area = len(pocket['atoms']) * 10  # Rough estimate
            volume = len(pocket['atoms']) * 8  # Rough estimate
        
        # Store properties
        pockets_with_properties[pocket_id] = pocket.copy()
        pockets_with_properties[pocket_id].update({
            'depth': max_depth,
            'surface_area': surface_area,
            'volume': volume
        })
    
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
    for identifier, atom_name, residue_name, chain_ID, residue_ID, x, y, z in PDB_iterator(file_path):
        residue_info = (residue_name, chain_ID, residue_ID)
        atom_to_residue[identifier] = residue_info
    
    for pocket_id, pocket in pockets_with_properties.items():
        residues = set()
        for atom_id in pocket['atoms']:
            if atom_id in atom_to_residue:
                residues.add(atom_to_residue[atom_id])
        
        # Store residue information
        pockets_with_residues[pocket_id] = pocket.copy()
        pockets_with_residues[pocket_id]['residues'] = list(residues)
    
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
            biochemically_filtered_pockets[pocket_id] = pocket.copy()
            biochemically_filtered_pockets[pocket_id]['biochemical_properties'] = biochemical_properties
    
    return biochemically_filtered_pockets

def run_pocket_detection(file_path, overlap_threshold=0.8):
    """
    Run the full pocket detection algorithm (steps 1-9)
    
    Returns:
    - final_pockets: Dictionary of filtered pockets that are likely to be active sites
    """
    # Steps 1-3 (already implemented in your code)
    # Get atom coordinates
    atoms_list, ids_list = get_atom_coordinates(file_path)
    
    # Create convex hull
    hull = create_convex_hull(file_path)
    
    # Steps 4-9 (implemented in this code)
    # Identify filled and empty voxels
    voxel_grid, atom_voxel_map, voxel_atom_map, hull = identify_filled_empty_voxels(file_path)
    
    # Define pockets from triangles
    pockets = define_pockets_from_triangles(hull, atoms_list, file_path)
    
    # Compute pocket overlap and merge pockets
    merged_pockets = compute_pocket_overlap(pockets, overlap_threshold)
    
    # Calculate pocket properties
    pockets_with_properties = calculate_pocket_properties(merged_pockets, file_path)
    
    # Identify pocket residues
    pockets_with_residues = identify_pocket_residues(pockets_with_properties, file_path)
    
    # Check biochemical conditions
    final_pockets = check_biochemical_conditions(pockets_with_residues, file_path)
    
    return final_pockets

# Example usage:
final_pockets = run_pocket_detection("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb")
print(f"Found {len(final_pockets)} potential active site pockets")

#run_pocket_detection("/home/xrs/projects-ubuntu/git_python/sbi_pyt_project/1a6u.pdb")