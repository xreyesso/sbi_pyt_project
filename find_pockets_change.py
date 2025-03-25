import sys
import math
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import argparse
import os
import logging
import json
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def PDB_iterator(pdb_file_path=None):
    """
    Iterate through atom records in PDB file.
    
    Args:
        pdb_file_path (str): Path to PDB file, reads from stdin if None
        
    Yields:
        tuple: Atom information
            (serial_number, atom_name, residue_name, chain_ID, residue_ID, x_coordinate, y_coordinate, z_coordinate)
    """
    try:
        if pdb_file_path is None:
            fi = sys.stdin
        else:
            if not os.path.exists(pdb_file_path):
                raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
            fi = open(pdb_file_path, "r")
        for line in fi:
            line = line.strip()
            if not line:
                continue
            record_name = line[0:6].strip()
            if (record_name == "ATOM"):
                try:
                    serial_number = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_ID = line[21].strip()
                    residue_ID = int(line[22:26].strip())
                    x_coordinate = float(line[30:38].strip())
                    y_coordinate = float(line[38:46].strip())
                    z_coordinate = float(line[46:54].strip())
                    yield (serial_number, atom_name, residue_name, chain_ID, residue_ID, x_coordinate, y_coordinate, z_coordinate)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing PDB line: {line}. Error: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error processing PDB file: {e}")
        raise
    finally:
        if pdb_file_path is not None and 'fi' in locals():
            fi.close()
def atoms_coordinates_dict(file_path):
    """
    Build atom ID to coordinates mapping from PDB file.
    
    Args:
        file_path (str): Path to PDB file
        
    Returns:
        dict: Mapping of atom IDs to coordinates (x,y,z)
    """
    atom_id_coords_dict = {}
    for atom_id, _, _, _, _, x_coordinate, y_coordinate, z_coordinate in PDB_iterator(file_path):
        atom_id_coords_dict[atom_id] = (x_coordinate, y_coordinate, z_coordinate)
    
    if not atom_id_coords_dict:
        logger.warning(f"No valid atoms found in file: {file_path}")
    
    return atom_id_coords_dict
def create_bounding_box_and_voxels(atom_coordinates, voxel_size=1.0):
    """
    Create bounding box containing all atoms and voxel grid.
    
    Args:
        atom_coordinates (dict): Atom ID to coordinates mapping
        voxel_size (float): Voxel size, default 1.0 Å
        
    Returns:
        tuple: (bounding box dict, voxel grid dict)
    """
    if not atom_coordinates:
        raise ValueError("Atom coordinates list is empty")
    
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
    bounding_box_dict = {"X": [xmin, xmax], "Y": [ymin, ymax], "Z": [zmin, zmax]}
    range_x = math.floor((xmax - xmin) / voxel_size) + 1
    range_y = math.floor((ymax - ymin) / voxel_size) + 1
    range_z = math.floor((zmax - zmin) / voxel_size) + 1
    logger.info(f"Voxel grid dimensions: x={range_x}, y={range_y}, z={range_z}")
    # Initialize voxel grid with default value to avoid explicit loops
    voxel_grid = defaultdict(lambda: False)
    return bounding_box_dict, voxel_grid
def create_convex_hull(atom_coordinates):
    """
    Create convex hull from atom coordinates.
    
    Args:
        atom_coordinates (dict): Atom ID to coordinates mapping
        
    Returns:
        tuple: (convex hull object, atom ID list)
    """
    if len(atom_coordinates) < 4:
        raise ValueError("At least 4 atom coordinates needed to create convex hull")
    
    atoms_list = []
    ids_list = []
    for id, coordinates in atom_coordinates.items():
        atoms_list.append((coordinates[0], coordinates[1], coordinates[2]))
        ids_list.append(id)
    
    atoms_np_array = np.array(atoms_list)
    try:
        hull = ConvexHull(atoms_np_array)
        return hull, ids_list
    except Exception as e:
        logger.error(f"Error creating convex hull: {e}")
        raise
def identify_filled_empty_voxels(atom_coordinates, voxel_size=1.0):
    """
    Identify filled and empty voxels.
    
    Args:
        atom_coordinates (dict): Atom ID to coordinates mapping
        voxel_size (float): Voxel size, default 1.0 Å
        
    Returns:
        tuple: (voxel grid, atom-to-voxel mapping, voxel-to-atom mapping)
    """
    box, voxel_grid = create_bounding_box_and_voxels(atom_coordinates, voxel_size)
    atom_voxel_map = {}
    voxel_atom_map = defaultdict(list)
    xmin = box["X"][0]
    ymin = box["Y"][0]
    zmin = box["Z"][0]
    for identifier, coordinates in atom_coordinates.items():
        i = math.floor((coordinates[0] - xmin) / voxel_size)
        j = math.floor((coordinates[1] - ymin) / voxel_size)
        k = math.floor((coordinates[2] - zmin) / voxel_size)
        
        voxel_idx = (i, j, k)
        voxel_grid[voxel_idx] = True
        atom_voxel_map[identifier] = voxel_idx
        voxel_atom_map[voxel_idx].append(identifier)
    
    return voxel_grid, atom_voxel_map, voxel_atom_map
def define_pockets_from_triangles(hull, ids_list, atoms_coordinates, box, voxel_grid, voxel_atom_map, voxel_size=1.0):
    """
    Define potential pockets based on convex hull triangles.
    
    Args:
        hull (ConvexHull): Convex hull object
        ids_list (list): Atom ID list
        atoms_coordinates (dict): Atom ID to coordinates mapping
        box (dict): Bounding box
        voxel_grid (dict): Voxel grid
        voxel_atom_map (dict): Voxel to atom mapping
        voxel_size (float): Voxel size, default 1.0 Å
        
    Returns:
        dict: Pockets dictionary with atoms, empty voxels, etc.
    """
    xmin, xmax = box["X"]
    ymin, ymax = box["Y"]
    zmin, zmax = box["Z"]
    pockets = {}
    triangle_ids = hull.simplices
    
    logger.info(f"Analyzing {len(triangle_ids)} triangular faces...")
    
    for idx, triangle in enumerate(triangle_ids):
        p1 = atoms_coordinates[ids_list[triangle[0]]]
        p2 = atoms_coordinates[ids_list[triangle[1]]]
        p3 = atoms_coordinates[ids_list[triangle[2]]]
        
        t_xmin = min(p1[0], p2[0], p3[0])
        t_xmax = max(p1[0], p2[0], p3[0])
        t_ymin = min(p1[1], p2[1], p3[1])
        t_ymax = max(p1[1], p2[1], p3[1])
        t_zmin = min(p1[2], p2[2], p3[2])
        t_zmax = max(p1[2], p2[2], p3[2])
        i_min = max(0, math.floor((t_xmin - xmin) / voxel_size))
        i_max = math.floor((t_xmax - xmin) / voxel_size) + 1
        j_min = max(0, math.floor((t_ymin - ymin) / voxel_size))
        j_max = math.floor((t_ymax - ymin) / voxel_size) + 1
        k_min = max(0, math.floor((t_zmin - zmin) / voxel_size))
        k_max = math.floor((t_zmax - zmin) / voxel_size) + 1
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
        # Calculate triangle normal vector
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        normal = np.cross(v1, v2)
        normal_length = np.linalg.norm(normal)
        
        # Avoid dividing by zero
        if normal_length > 0:
            normal = normal / normal_length
        else:
            logger.warning(f"Triangle {idx} has zero normal vector")
            normal = np.array([0, 0, 0])
        pockets[idx] = {
            'atoms': set(pocket_atoms),
            'empty_voxels': set(pocket_empty_voxels),
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
    Compute overlap between pockets and merge highly overlapping ones.
    
    Args:
        pockets (dict): Pockets dictionary
        overlap_threshold (float): Overlap threshold, default 0.8
        
    Returns:
        dict: Merged pockets dictionary
    """
    if not pockets:
        logger.warning("No pockets to merge")
        return {}
    
    overlap_matrix = {}
    for p1_id in pockets:
        overlap_matrix[p1_id] = {}
        for p2_id in pockets:
            if p1_id != p2_id:
                p1_atoms = pockets[p1_id]['atoms']
                p2_atoms = pockets[p2_id]['atoms']
                
                # Avoid dividing by zero
                if len(p1_atoms) == 0 or len(p2_atoms) == 0:
                    overlap = 0
                else:
                    overlap = len(p1_atoms.intersection(p2_atoms)) / min(len(p1_atoms), len(p2_atoms))
                
                overlap_matrix[p1_id][p2_id] = overlap
    merged = {}
    merged_pockets = {}
    next_merged_id = 0
    
    for p1_id in pockets:
        if p1_id in merged:
            continue
            
        merged_pocket = {
            'atoms': pockets[p1_id]['atoms'].copy(),  # Use copy to avoid modifying original data
            'constituent_pockets': [p1_id]
        }
        merged[p1_id] = next_merged_id
        pockets_merged_this_round = [p1_id]
        while pockets_merged_this_round:
            next_id = pockets_merged_this_round.pop()
        
            for p2_id in pockets:
                if p2_id not in merged:
                    if overlap_matrix[next_id][p2_id] >= overlap_threshold or overlap_matrix[p2_id][next_id] >= overlap_threshold:
                        merged[p2_id] = next_merged_id
                        merged_pocket['atoms'].update(pockets[p2_id]['atoms'])
                        merged_pocket['constituent_pockets'].append(p2_id)
                        pockets_merged_this_round.append(p2_id)
        
        merged_pockets[next_merged_id] = merged_pocket
        next_merged_id += 1
    
    logger.info(f"Merged {len(pockets)} initial pockets into {len(merged_pockets)} pockets")
    return merged_pockets
def calculate_pocket_properties(merged_pockets, atoms_coordinates):
    """
    Calculate geometric properties of pockets: volume, surface area, and depth.
    
    Args:
        merged_pockets (dict): Merged pockets dictionary
        atoms_coordinates (dict): Atom ID to coordinates mapping
        
    Returns:
        dict: Pockets dictionary with geometric properties
    """
    pockets_with_properties = {}
    
    for pocket_id, pocket in merged_pockets.items():
        max_depth = 0
        pocket_atoms_coords = []
        
        # Collect coordinates of all atoms in the pocket
        for atom_id in pocket['atoms']:
            if atom_id in atoms_coordinates:  # Ensure atom ID exists
                pocket_atoms_coords.append(atoms_coordinates[atom_id])
        
        # Pocket must have at least 4 atoms to calculate geometric properties
        if len(pocket_atoms_coords) < 4:
            logger.warning(f"Pocket {pocket_id} has insufficient atoms ({len(pocket_atoms_coords)} < 4), skipping")
            continue
        
        # Calculate pocket centroid
        centroid = np.mean(pocket_atoms_coords, axis=0)
        
        # Calculate depth
        for atom_coord in pocket_atoms_coords:
            distance = np.linalg.norm(np.array(atom_coord) - centroid)
            max_depth = max(max_depth, distance)
        try:
            # Calculate surface area and volume
            pocket_hull = ConvexHull(np.array(pocket_atoms_coords))
            surface_area = pocket_hull.area
            volume = pocket_hull.volume
            
            pockets_with_properties[pocket_id] = {
                **pocket,
                'depth': max_depth,
                'surface_area': surface_area,
                'volume': volume,
                'centroid': centroid.tolist(),  # Convert to list for JSON serialization
                'atom_count': len(pocket['atoms'])
            }
        except Exception as e:
            logger.warning(f"Error calculating geometric properties for pocket {pocket_id}: {e}")
    
    return pockets_with_properties
def identify_pocket_residues(pockets_with_properties, file_path):
    """
    Identify amino acid residues in pockets.
    
    Args:
        pockets_with_properties (dict): Pockets dictionary with geometric properties
        file_path (str): Path to PDB file
        
    Returns:
        dict: Pockets dictionary with residue information
    """
    pockets_with_residues = {}
    
    # Create atom ID to residue mapping
    atom_to_residue = {}
    for identifier, atom_name, residue_name, chain_ID, residue_ID, _, _, _ in PDB_iterator(file_path):
        residue_info = (residue_name, chain_ID, residue_ID, atom_name)
        atom_to_residue[identifier] = residue_info
    
    for pocket_id, pocket in pockets_with_properties.items():
        residues = set()
        residue_atom_counts = defaultdict(int)
        
        for atom_id in pocket['atoms']:
            if atom_id in atom_to_residue:
                residue_name, chain_ID, residue_ID, atom_name = atom_to_residue[atom_id]
                residue_key = (residue_name, chain_ID, residue_ID)
                residues.add(residue_key)
                residue_atom_counts[residue_key] += 1
        
        # Store residue information
        pockets_with_residues[pocket_id] = {
            **pocket,
            'residues': list(residues),  # Convert to list for JSON serialization
            'residue_count': len(residues),
            'residue_atom_counts': dict(residue_atom_counts)  # Convert to regular dict for JSON serialization
        }
    
    return pockets_with_residues
def check_biochemical_conditions(pockets_with_residues):
    """
    Check biochemical conditions of pockets, filter potential active sites.
    
    Args:
        pockets_with_residues (dict): Pockets dictionary with residue information
        
    Returns:
        dict: Pockets dictionary filtered by biochemical conditions
    """
    # Define residues and atoms for hydrogen bond acceptors/donors and van der Waals interactions
    hydrogen_bond_acceptors = {'GLN', 'ASN', 'TYR', 'SER', 'THR', 'HIS', 'ASP', 'GLU'}
    hydrogen_bond_donors = {'THR', 'SER', 'TYR', 'LYS', 'ARG', 'HIS', 'TRP'}
    hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR'}
    charged = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS'}
    sulfur_interactions = {'CYS', 'MET'}
    ion_interactions = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS'}
    vdw_interactions = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'PRO', 'TRP', 'TYR'}
    proline_vdw = {'PRO'}
    nitrogen_based = {'GLN', 'ASN', 'TYR', 'HIS', 'TRP'}
    no_participation = {'GLY'}

    biochemical_filtered_pockets = {}
    
    for pocket_id, pocket in pockets_with_residues.items():
        # Count each type of residue
        ha_count = 0
        hd_count = 0
        hydrophobic_count = 0
        charged_count = 0
        sulfur_count = 0
        ion_count = 0
        vdw_count = 0
        proline_count = 0
        nitrogen_based_count = 0
      
        residue_types = {
            'hydrogen_acceptors': [],
            'hydrogen_donors': [],
            'hydrophobic': [],
            'charged': [],
            'sulfur_interactions': [],
            'ion_interactions': [],
            'vdw_interactions': [],
            'proline_vdw': [],
            'nitrogen_based': [],
            'no_participation': []
        }
        for residue_info in pocket['residues']:
            residue_name, chain_ID, residue_ID = residue_info
            
            if residue_name in hydrogen_bond_acceptors:
                ha_count += 1
                residue_types['hydrogen_acceptors'].append(f"{residue_name}{residue_ID}{chain_ID}")
            
            if residue_name in hydrogen_bond_donors:
                hd_count += 1
                residue_types['hydrogen_donors'].append(f"{residue_name}{residue_ID}{chain_ID}")
            
            if residue_name in hydrophobic:
                hydrophobic_count += 1
                residue_types['hydrophobic'].append(f"{residue_name}{residue_ID}{chain_ID}")
            
            if residue_name in charged:
                charged_count += 1
                residue_types['charged'].append(f"{residue_name}{residue_ID}{chain_ID}")

            if residue_name in sulfur_interactions:
                sulfur_count += 1
                residue_types['sulfur_interactions'].append(f"{residue_name}{residue_ID}{chain_ID}")
        
            if residue_name in ion_interactions:
                ion_count += 1
                residue_types['ion_interactions'].append(f"{residue_name}{residue_ID}{chain_ID}")
        
            if residue_name in vdw_interactions:
                vdw_count += 1
                residue_types['vdw_interactions'].append(f"{residue_name}{residue_ID}{chain_ID}")
        
            if residue_name in proline_vdw:
                proline_count += 1
                residue_types['proline_vdw'].append(f"{residue_name}{residue_ID}{chain_ID}")
        
            if residue_name in nitrogen_based:
                nitrogen_based_count += 1
                residue_types['nitrogen_based'].append(f"{residue_name}{residue_ID}{chain_ID}")
        
            if residue_name in no_participation:
                residue_types['no_participation'].append(f"{residue_name}{residue_ID}{chain_ID}")
        
        biochemical_properties = {
            'hydrogen_acceptor_count': ha_count,
            'hydrogen_donor_count': hd_count,
            'hydrophobic_count': hydrophobic_count,
            'charged_count': charged_count,
            'sulfur_count': sulfur_count,
            'ion_count': ion_count,
            'vdw_count': vdw_count,
            'proline_count': proline_count,
            'nitrogen_based_count': nitrogen_based_count,
            'residue_types': residue_types
        }
        
        # Filter condition: at least 1 hydrogen bond acceptor, 1 donor, and 2 hydrophobic residues
        if ha_count > 0 and hd_count > 0 and hydrophobic_count >= 2:
            biochemical_filtered_pockets[pocket_id] = {
                **pocket,
                'biochemical_properties': biochemical_properties
            }
    
    logger.info(f"After biochemical filtering, {len(biochemical_filtered_pockets)} pockets retained")
    return biochemical_filtered_pockets
    
def rank_pockets(pockets):
    """
    Rank pockets based on volume, surface area, residue diversity and biochemical properties.
    
    Args:
        pockets (dict): Pockets dictionary
        
    Returns:
        list: Ranked list of pocket IDs
    """
    if not pockets:
        return []
    
    # Calculate composite score for each pocket
    scores = {}
    
    for pocket_id, pocket in pockets.items():
        # Volume and surface area scores
        volume_score = pocket['volume'] / 100
        surface_score = pocket['surface_area'] / 100
        
        # Residue diversity score
        residue_diversity = len(set(res[0] for res in pocket['residues']))
        diversity_score = residue_diversity / 5
        
        # Biochemical properties score
        bio_score = (
            pocket['biochemical_properties']['hydrogen_acceptor_count'] +
            pocket['biochemical_properties']['hydrogen_donor_count'] +
            pocket['biochemical_properties']['hydrophobic_count'] * 0.5 +
            pocket['biochemical_properties']['charged_count'] * 0.5 +
            pocket['biochemical_properties']['sulfur_count'] * 0.5 +  
            pocket['biochemical_properties']['ion_count'] * 0.5 +     
            pocket['biochemical_properties']['vdw_count'] * 0.3 +      
            pocket['biochemical_properties']['proline_count'] * 0.3 +  
            pocket['biochemical_properties']['nitrogen_based_count'] * 0.4  
        )
        
        # Composite score
        total_score = volume_score + surface_score + diversity_score + bio_score
        scores[pocket_id] = total_score
    
    # Sort pocket IDs by score
    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    return ranked_ids
def visualize_pockets(pockets, atoms_coordinates, output_prefix):
    """
    Generate 3D visualizations of pockets.
    
    Args:
        pockets (dict): Pockets dictionary
        atoms_coordinates (dict): Atom ID to coordinates mapping
        output_prefix (str): Output file prefix
    """
    if not pockets:
        logger.warning("No pockets to visualize")
        return
    
    # Visualize only top 5 largest pockets
    ranked_ids = rank_pockets(pockets)
    top_pockets = ranked_ids[:min(5, len(ranked_ids))]
    
    for pocket_id in top_pockets:
        pocket = pockets[pocket_id]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms in pocket
        xs, ys, zs = [], [], []
        for atom_id in pocket['atoms']:
            if atom_id in atoms_coordinates:
                x, y, z = atoms_coordinates[atom_id]
                xs.append(x)
                ys.append(y)
                zs.append(z)
        
        ax.scatter(xs, ys, zs, c='r', marker='o', label='Pocket Atoms')
        
        # Plot pocket centroid
        centroid = pocket['centroid']
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c='g', marker='*', s=200, label='Centroid')
        
        # Set figure title and labels
        ax.set_title(f'Pocket {pocket_id} (Volume: {pocket["volume"]:.2f}, Residues: {pocket["residue_count"]})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # Save image
        plt.savefig(f'{output_prefix}_pocket_{pocket_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated visualization for pocket {pocket_id}")
def export_results(pockets, output_prefix):
    """
    Export results as JSON and text report formats.
    
    Args:
        pockets (dict): Pockets dictionary
        output_prefix (str): Output file prefix
    """
    if not pockets:
        logger.warning("No pockets to export")
        return
    
    # Create JSON-serializable copy
    json_serializable_pockets = {}
    
    for pocket_id, pocket in pockets.items():
        serializable_pocket = {}
        
        # Process each key-value pair, ensuring all are JSON-serializable
        for key, value in pocket.items():
            if key == 'residues':
                # Convert residue tuples to string list
                serializable_pocket[key] = [f"{r[0]}_{r[1]}_{r[2]}" for r in value]
            elif key == 'residue_atom_counts':
                # Convert tuple-keyed dict to string-keyed dict
                serializable_counts = {}
                for res_tuple, count in value.items():
                    res_key = f"{res_tuple[0]}_{res_tuple[1]}_{res_tuple[2]}"
                    serializable_counts[res_key] = count
                serializable_pocket[key] = serializable_counts
            elif isinstance(value, set):
                # Convert sets to lists
                serializable_pocket[key] = list(value)
            elif key == 'biochemical_properties' and isinstance(value, dict):
                # Handle nested biochemical_properties dict
                serializable_bio = {}
                for bio_key, bio_value in value.items():
                    if isinstance(bio_value, set):
                        serializable_bio[bio_key] = list(bio_value)
                    elif bio_key == 'residue_types' and isinstance(bio_value, dict):
                        serializable_rt = {}
                        for rt_key, rt_value in bio_value.items():
                            if isinstance(rt_value, set):
                                serializable_rt[rt_key] = list(rt_value)
                            else:
                                serializable_rt[rt_key] = rt_value
                        serializable_bio[bio_key] = serializable_rt
                    else:
                        serializable_bio[bio_key] = bio_value
                serializable_pocket[key] = serializable_bio
            else:
                serializable_pocket[key] = value
        
        json_serializable_pockets[pocket_id] = serializable_pocket
    
    # Export full results as JSON
    with open(f'{output_prefix}_full_results.json', 'w') as f:
        json.dump(json_serializable_pockets, f, indent=2)
    
    # Create concise text report
    ranked_ids = rank_pockets(pockets)
    
    with open(f'{output_prefix}_report.txt', 'w') as f:
        f.write(f"Found {len(pockets)} potential active pockets\n\n")
        
        for rank, pocket_id in enumerate(ranked_ids, 1):
            pocket = pockets[pocket_id]
            
            f.write(f"Pocket {pocket_id} (Rank #{rank}):\n")
            f.write(f"  Volume: {pocket['volume']:.2f} Å³\n")
            f.write(f"  Surface Area: {pocket['surface_area']:.2f} Å²\n")
            f.write(f"  Residue Count: {pocket['residue_count']}\n")
            f.write(f"  Atom Count: {pocket['atom_count']}\n")
            
            # Add biochemical properties information
            bio = pocket['biochemical_properties']
            f.write("  Biochemical Properties:\n")
            f.write(f"    Hydrogen Bond Acceptors: {bio['hydrogen_acceptor_count']}\n")
            f.write(f"    Hydrogen Bond Donors: {bio['hydrogen_donor_count']}\n")
            f.write(f"    Hydrophobic Residues: {bio['hydrophobic_count']}\n")
            f.write(f"    Charged Residues: {bio['charged_count']}\n")
            f.write(f"    Sulfur Interactions: {bio.get('sulfur_count', 0)}\n") 
            f.write(f"    Ion Interactions: {bio.get('ion_count', 0)}\n")      
            f.write(f"    Van der Waals Interactions: {bio.get('vdw_count', 0)}\n")  
            f.write(f"    Proline-specific Interactions: {bio.get('proline_count', 0)}\n")  
            f.write(f"    Nitrogen-based Interactions: {bio.get('nitrogen_based_count', 0)}\n") 
            
            # List residues in pocket
            f.write("  Residue List:\n")
            for i, res in enumerate(pocket['residues'], 1):
                res_name, chain, res_id = res
                f.write(f"    {i}. {res_name} {res_id} {chain}\n")
            
            f.write("\n")
    
    logger.info(f"Results exported to {output_prefix}_full_results.json and {output_prefix}_report.txt")
def run_pocket_detection(file_path, overlap_threshold=0.8, voxel_size=1.0, output_prefix=None):
    """
    Run complete pocket detection workflow.
    
    Args:
        file_path (str): Path to PDB file
        overlap_threshold (float): Overlap threshold, default 0.8
        voxel_size (float): Voxel size, default 1.0 Å
        output_prefix (str): Output file prefix, default None
        
    Returns:
        dict: Final detected pockets dictionary
    """
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(file_path))[0]
    
    logger.info(f"Starting processing of {file_path}")
    logger.info(f"Parameters: overlap_threshold={overlap_threshold}, voxel_size={voxel_size}")
    
    try:
        # Step 1: Read atom coordinates
        atoms_ids_and_coordinates = atoms_coordinates_dict(file_path)
        logger.info(f"Read {len(atoms_ids_and_coordinates)} atoms")
        # Step 2: Create convex hull
        hull, ids_list = create_convex_hull(atoms_ids_and_coordinates)
        logger.info(f"Created convex hull with {len(hull.simplices)} faces")
        # Step 3: Identify filled and empty voxels
        voxel_grid, atom_voxel_map, voxel_atom_map = identify_filled_empty_voxels(atoms_ids_and_coordinates, voxel_size)
        
        # Step 4: Create bounding box
        box, _ = create_bounding_box_and_voxels(atoms_ids_and_coordinates, voxel_size)
        
        # Step 5: Define potential pockets
        initial_pockets = define_pockets_from_triangles(hull, ids_list, atoms_ids_and_coordinates, box, voxel_grid, voxel_atom_map, voxel_size)
        logger.info(f"Identified {len(initial_pockets)} initial pockets")
        
        # Step 6: Merge overlapping pockets
        merged_pockets = compute_pocket_overlap(initial_pockets, overlap_threshold)
        logger.info(f"{len(merged_pockets)} pockets remain after merging")
    
        # Step 7: Calculate pocket properties
        pockets_with_properties = calculate_pocket_properties(merged_pockets, atoms_ids_and_coordinates)
        logger.info(f"Successfully calculated geometric properties for {len(pockets_with_properties)} pockets")
    
        # Step 8: Identify pocket residues
        pockets_with_residues = identify_pocket_residues(pockets_with_properties, file_path)
        logger.info(f"Successfully identified residue information for {len(pockets_with_residues)} pockets")
    
        # Step 9: Apply biochemical condition filtering
        final_pockets = check_biochemical_conditions(pockets_with_residues)
        logger.info(f"{len(final_pockets)} pockets remain after biochemical filtering")
    
        # Step 10: Visualize pockets
        visualize_pockets(final_pockets, atoms_ids_and_coordinates, output_prefix)
    
        # Step 11: Export results
        export_results(final_pockets, output_prefix)
    
        return final_pockets

    except Exception as e:
        logger.error(f"Error during pocket detection: {e}")
        raise

def main():
    """
    Main function to handle command line arguments and run pocket detection.
    """
    parser = argparse.ArgumentParser(description='Protein pocket detection tool')
    parser.add_argument('pdb_file', type=str, help='Path to input PDB file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file prefix')
    parser.add_argument('-t', '--threshold', type=float, default=0.8, help='Pocket overlap merging threshold (0.0-1.0)')
    parser.add_argument('-v', '--voxel', type=float, default=1.0, help='Voxel size (Å)')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed log output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate parameters
    if not os.path.exists(args.pdb_file):
        logger.error(f"PDB file does not exist: {args.pdb_file}")
        sys.exit(1)
    
    if args.threshold < 0.0 or args.threshold > 1.0:
        logger.warning(f"Overlap threshold should be in the range of 0.0-1.0, current value: {args.threshold}")
    
    if args.voxel <= 0:
        logger.error(f"Voxel size must be a positive number, current value: {args.voxel}")
        sys.exit(1)
    
    try:
        # Run pocket detection
        pockets = run_pocket_detection(
            args.pdb_file,
            overlap_threshold=args.threshold,
            voxel_size=args.voxel,
            output_prefix=args.output
        )
        
        # Print results summary
        if pockets:
            ranked_ids = rank_pockets(pockets)
            print(f"\nDetected {len(pockets)} potential active pockets")
            
            for rank, pocket_id in enumerate(ranked_ids[:3], 1):
                pocket = pockets[pocket_id]
                print(f"\nPocket {pocket_id} (Rank #{rank}):")
                print(f"  Volume: {pocket['volume']:.2f} Å³")
                print(f"  Surface Area: {pocket['surface_area']:.2f} Å²")
                print(f"  Residue Count: {pocket['residue_count']}")
                print(f"  Atom Count: {pocket['atom_count']}")
            
            if len(ranked_ids) > 3:
                print(f"\nThere are {len(ranked_ids) - 3} more pockets. See output file for details.")
        else:
            print("No active pockets detected")
            
    except Exception as e:
        logger.error(f"Program execution error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


