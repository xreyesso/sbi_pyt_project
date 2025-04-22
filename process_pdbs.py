import os
import sys
import re
from chimera import runCommand as rc

def process_pdb_with_ligand(input_pdb, output_dir, distance=4.0):
    """
    Process a PDB file with ligand, to extract the residues near the ligand.
    """
    # Create an output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the full output path for the PDB file
    base_name = os.path.basename(input_pdb).split('.')[0]
    output_path = os.path.join(output_dir, base_name + "_nearby_residues.pdb")
    
    # Open the PDB file in Chimera
    rc("open " + input_pdb)
    
    # Select the ligand
    rc("select ligand")
    
    # Select residues within a specific distance of the ligand
    rc("select sel za<{0} & ~sel".format(distance))
    
    # Write the selected residues into a new PDB file (the output path created beforehand)
    rc("write selected #0 " + output_path)
    
    # Close all models
    rc("close all")
    
    print("\n")
    return output_path

def process_all_pdbs(input_dir, output_dir, distance=4.0):
    """
    Process all the PDB files in a directory.
    """
    processed_files = []
    
    # OBtain all PDB files in the input directory
    pdb_files = [file for file in os.listdir(input_dir) if file.endswith('.pdb')]
    
    # For each pdb file call the preocess pdb with ligand function
    for pdb_file in pdb_files:
        input_path = os.path.join(input_dir, pdb_file)
        output_path = process_pdb_with_ligand(input_path, output_dir, distance)
        processed_files.append((pdb_file, output_path))
    
    return processed_files

def extract_residues_from_pdb(pdb_file):
    """
    Extract residue names and IDs from the PDB file. Only includiing the 
    ATOM lines (not HOH or other HETATM).
    """
    # We create a set strucutre as a lot of residues tend to be repeated due 
    # to multiple atoms being closeby
    residues = set()
    
    with open(pdb_file, 'r') as fi:
        for line in fi:
            # If lines starts with ATOM
            if line.startswith("ATOM"):
                # Extract residue name and ID
                residue_name = line[17:20].strip()
                residue_ID = line[22:26].strip()
                
                # Combine residue name and ID
                full_residue_id = residue_name + residue_ID
                residues.add(full_residue_id)
    
    # Convert to sorted list by the residue id
    # Finds the digits inside the string, joins them and turns them into an integer
    residue_list = sorted(list(residues), key=lambda x: int(''.join(filter(str.isdigit, x))))
    return residue_list

def create_residue_summary(processed_files, output_dir):
    """
    Create a summary file with the PDB names and the residues close to the ligand.
    """
    summary_path = os.path.join(output_dir, "residue_summary.txt")
    
    with open(summary_path, 'w') as f:
        for original_pdb, output_pdb in processed_files:
            # Extract residues from the output PDB
            residues = extract_residues_from_pdb(output_pdb)
            residue_str = ", ".join(residues)
            f.write("{0}: {1}\n".format(original_pdb, residue_str))
    
    return summary_path

if __name__ == "__main__":
    # If there's less than 3 command line arguments exit the program
    if len(sys.argv) < 3:
        print("chimera --nogui --script \"process_pdbs.py . ./nearby_residues 4.0\"")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    distance = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0 # if the distance is not included use 4 as default
    
    processed_files = process_all_pdbs(input_dir, output_dir, distance)
    summary_path = create_residue_summary(processed_files, output_dir)
    
    print("Processed {0} PDB files. Results saved in {1}".format(len(processed_files), output_dir))
    print("Residue summary saved to {0}".format(summary_path))