import pandas as pd

# Input file paths
complex_size_file = "/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/complex_size.tsv"
binding_affinity_file = "/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/binding_affinity_two_proteins.tsv"

# Output file path
output_file = "/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/binding_affinity_two_protein.tsv"

# Load the data
df_size = pd.read_csv(complex_size_file, sep="\t")
df_affinity = pd.read_csv(binding_affinity_file, sep="\t")

# Merge on PDB_ID (inner join keeps only common IDs)
df_merged = pd.merge(df_affinity, df_size, on="PDB_ID", how="inner")

# Select and rename the desired columns
df_final = df_merged[["PDB_ID", "ΔG_kJ/mol", "Complex_Count"]].rename(
    columns={"PDB_ID": "pdb_id", "ΔG_kJ/mol": "delta_G", "Complex_Count": "complex_count"}
)

# Save to CSV
df_final.to_csv(output_file, index=False)

print(f"Saved merged data to {output_file}")
