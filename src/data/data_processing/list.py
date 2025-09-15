import pandas as pd

# Input and output paths
input_file = "/home/op98/protein_design/dataset/PP/PP_anti/filtered_file1.csv"
output_file = "/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/ids.txt"

# Read CSV
df = pd.read_csv(input_file)

# Extract the ID column
pdb_ids = df["ID"]

# Save to txt (one per line, no header/index)
pdb_ids.to_csv(output_file, index=False, header=False)

print(f"PDB IDs saved to: {output_file}")
