import os
import re
import sys
import subprocess
from collections import defaultdict

def parse_residues_file(file_path):
    """
    Parse nearby_residues summary file (has nearby residues of ligand for each PDB).
    """
    residues_dict = {}
    
    with open(file_path, 'r') as fi:
        for line in fi:
            line = line.strip()
            
            # if it's empty skip
            if not line:
                continue

            # split line by :    
            parts = line.split(':')
            if len(parts) != 2:
                continue
                
            pdb_id = parts[0].lower()
            residue_str = parts[1].strip()
            
            # Iterate through the comma separated residues
            residues = []
            for residue in residue_str.split(','):
                residue = residue.strip()
                # Split residue into name and numbers
                match = re.match(r'([A-Z]+)(\d+)', residue)
                if match:
                    residue_name, residue_ID = match.groups()
                    residues.append((residue_name, residue_ID))
            
            residues_dict[pdb_id] = residues
    
    return residues_dict

def parse_pdb_mapping(mapping_file):
    """
    Parse mapping file between the liganded and unliganded PDB files.
    """
    pdb_mapping = {}
    
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Divide the line by the space    
            parts = line.split()
            if len(parts) == 2:
                liganded_pdb = parts[0].lower()
                unliganded_pdb = parts[1].lower()
                
                # Add the .pdb extension if it is not there
                if not liganded_pdb.endswith('.pdb'):
                    liganded_pdb += '.pdb'
                if not unliganded_pdb.endswith('.pdb'):
                    unliganded_pdb += '.pdb'
                    
                pdb_mapping[unliganded_pdb] = liganded_pdb
    
    return pdb_mapping

def run_find_pockets(pdb_file, output_dir):
    """
    Run the find pockets program on the unbound PDB file.
    """
    # Create output directory for this PDB
    pdb_name = os.path.basename(pdb_file)
    protein = pdb_name.split('.')[0]
    pdb_output_dir = os.path.join(output_dir, f"{protein}_output")
    
    # Check if the pocket detector is available an import the function
    try:
        # Import the main function (the file should be in the same directory)
        from find_pockets import run_complete_workflow
        run_complete_workflow(pdb_file, pdb_output_dir)
    except ImportError:
        # If import fails give an error
        print("Error: find pockets program is not in this folder")
        sys.exit(1)
    
    return pdb_output_dir

def parse_pocket_residues(residues_report):
    """
    Parse the pocket_report.txt file to get residues for each pocket.
    """
    pocket_residues = defaultdict(list)
    current_pocket = None
    
    try:
        with open(residues_report, 'r') as fi:
            for line in fi:
                line = line.strip()
                
                # Find pocket number
                if line.startswith("==== POCKET"):
                    pocket_num = int(line.split()[2])
                    current_pocket = pocket_num
                
                # If current pocket exists and the line starts with "Residues in the surface of the pocket"
                elif current_pocket and "Residues in the surface of the pocket" in line:
                    # Next line will contain the residues
                    residue_line = next(fi).strip()
                    
                    # Split the residues
                    for residue in residue_line.split(', '):
                        # Format: A-VAL183 -> split into chain and residue part
                        parts = residue.split('-')
                        if len(parts) == 2:
                            chain = parts[0]
                            # Extract residue name and ID
                            match = re.match(r'([A-Z]+)(\d+)', parts[1])
                            if match:
                                residue_name, residue_ID = match.groups()
                                pocket_residues[current_pocket].append((residue_name, residue_ID))
                    
    except Exception as e:
        print(f"Error parsing pocket residues from {residues_report}")
    
    return pocket_residues

def calculate_match_percentage(known_residues, pocket_residues):
    """
    Calculate what percentage of known residues nearby the ligand
    are in the pocket.
    """
    if not known_residues:
        return 0, set()
    
    # Convert lists to sets for easier comparison
    known_set = set((name, ID) for name, ID in known_residues)
    pocket_set = set((name, ID) for name, ID in pocket_residues)
    
    # Find intersection
    matches = known_set.intersection(pocket_set)
    
    # Calculate percentage
    percentage = (len(matches) / len(known_set)) * 100
    
    return percentage, matches

# Main function
def main():
    if len(sys.argv) < 5:
        print("Usage: python pocket_validator.py <nearby_residues_file> <pdb_mapping_file> <pdb_directory> <output_directory>")
        sys.exit(1)
    
    residues_file = sys.argv[1]
    pdb_mapping_file = sys.argv[2]
    pdb_dir = sys.argv[3]
    output_dir = sys.argv[4]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtain known residues nearby the ligand of the different pdbs
    known_residues = parse_residues_file(residues_file)
    
    # Parse PDB mapping
    pdb_mapping = parse_pdb_mapping(pdb_mapping_file)
    
    # Create results file
    results_file = os.path.join(output_dir, 'pocket_detection_results.txt')
    
    with open(results_file, 'w') as fi:
        # Updated header with new columns
        fi.write("PDB File\tLiganded PDB\tHit/Miss\tBest Match Percentage\tBest Matching Pocket\tMatched Residues\n")
        
        # Process each unliganded PDB file
        for unliganded_pdb in pdb_mapping:
            liganded_pdb = pdb_mapping[unliganded_pdb]
            
            # Check if we have known residues for the liganded PDB
            if liganded_pdb not in known_residues:
                print(f"Error: No known residues for {liganded_pdb}.")
                continue
                
            print(f"Processing {unliganded_pdb} (corresponding to {liganded_pdb})")
            
            # Find the unliganded PDB file in the directory
            pdb_path = None
            for file in os.listdir(pdb_dir):
                if file.lower() == unliganded_pdb.lower():
                    pdb_path = os.path.join(pdb_dir, file)
                    break
                    
            if not pdb_path:
                print(f"Error: Could not find {unliganded_pdb} in {pdb_dir}.")
                continue
            
            # Run find pocket on the unliganded PDB
            pocket_output_dir = run_find_pockets(pdb_path, output_dir)
            if not pocket_output_dir:
                continue
                
            # Obtain pocket residues from the find pocket program results
            residues_report = os.path.join(pocket_output_dir, 'pockets_report.txt')
            if not os.path.exists(residues_report):
                print(f"Error: Pocket residues repor not found for {unliganded_pdb}")
                continue
            pocket_residues = parse_pocket_residues(residues_report)
            
            # Calculate match percentages for the top 5 pockets
            best_match = 0
            best_pocket = None
            best_matches = set()
            
            # Get top 5 pockets
            top_pockets = sorted(pocket_residues.keys())[:5] if pocket_residues else []
            
            for pocket_num in top_pockets:
                match_percentage, matched_residues = calculate_match_percentage(
                    known_residues[liganded_pdb], pocket_residues[pocket_num])
                
                print(f"  Pocket {pocket_num}: {match_percentage:.2f}% match")
                
                if match_percentage > best_match:
                    best_match = match_percentage
                    best_pocket = pocket_num
                    best_matches = matched_residues
            
            # Determine if this is a hit (≥50% match in any of the top 5 pockets)
            score = 1 if best_match >= 50 else 0
            
            # Format matched residues string in the requested format
            matched_residues_str = f"Pocket {best_pocket}: {', '.join([f'{res[0]}{res[1]}' for res in best_matches])}"
            
            # Write result to file with the new column
            fi.write(f"{unliganded_pdb}\t{liganded_pdb}\t{score}\t{best_match:.2f}%\t{best_pocket}\t{matched_residues_str}\n")
            
            print(f"  Result for {unliganded_pdb}: {'HIT' if best_match >= 50 else 'MISS'} (best match: {best_match:.2f}% in pocket {best_pocket})")
            print(f"  Matched residues: {', '.join([f'{res[0]}{res[1]}' for res in best_matches])}\n")
    
    print(f"Results written to {results_file}")

if __name__ == "__main__":
    main()