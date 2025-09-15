import torch

data = torch.load("/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/test_data/1a2k_charged.pt")
N = data["coords"].shape[0]

print(data.keys())  # should include 'chain'

# sanity check: all fields align
assert len(data["chain"]) == N == len(data["resname"]) == len(data["resseq"]) == len(data["icode"]) == len(data["group_id"])

for i in range(N):
    chain = data["chain"][i]
    # blank chain IDs exist in some PDBs; make them visible:
    if chain == "" or chain == " ":
        chain = "(blank)"
    res_id = f"{data['resseq'][i]}{data['icode'][i]}" if data['icode'][i] else str(data['resseq'][i])
    print(
        f"{i:>3}: {data['resname'][i]:>3}  "
        f"charge={int(data['charge'][i])}  "
        f"chain={chain}  res={res_id}  "
        f"group_id={data['group_id'][i]}  "
        f"coords={data['coords'][i].tolist()}"
    )
