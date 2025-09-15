#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:08:23 2023

@author: shenli
"""
import Bio.PDB as PDB
from Bio.PDB.Polypeptide import is_aa
import numpy as np
import pandas
import math
import csv

def Gauss_linking_integral(line1,line2):
    '''
    a:tuple; elements are head and tail
    of line segment, each is a (3,) array representing the xyz coordinate.
    '''
    #a0, a1 = a[0],a[1]
    #b0, b1 = b[0],b[1]
    a = [line1.startpoint,line1.endpoint]
    b = [line2.startpoint,line2.endpoint]
    
    R = np.empty((2,2),dtype=tuple)
    for i in range(2):
        for j in range(2):
            R[i,j]=a[i]-b[j]
          
    n=[]
    cprod = []

    cprod.append(np.cross(R[0,0],R[0,1]))
    cprod.append(np.cross(R[0,1],R[1,1]))
    cprod.append(np.cross(R[1,1],R[1,0]))
    cprod.append(np.cross(R[1,0],R[0,0]))
    
    for c in cprod:
        n.append(c/(np.linalg.norm(c)+1e-6))
    
    area1 = np.arcsin(np.dot(n[0],n[1]))
    area2 = np.arcsin(np.dot(n[1],n[2]))
    area3 = np.arcsin(np.dot(n[2],n[3]))
    area4 = np.arcsin(np.dot(n[3],n[0]))
    
    sign = np.sign(np.cross(a[1]-a[0],b[1]-b[0]).dot(a[0]-b[0]))
    Area = area1+area2+area3+area4

    return sign*Area


class Line():
    def __init__(self, startpoint, endpoint, start_type, end_type):
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.start_type = start_type
        self.end_type = end_type

class Atom():
    def __init__(self, data, etype, eid=None):  # data is (3,) array, etype is str
        self.data = data
        self.etype = etype # atom element
        self.eid = eid #atom full name

class bonded_Atom():
    def __init__(self, lines, atom):  # (init with a list of line object)
        self.lines = lines
        self.atom = atom

    def add_line(self, line):
        self.lines.append(line)

def get_bond(atoms,atom):
    maximal_bond_lengths = [["C","H",1.19],
                            ["C","C",1.64],
                            ["C","S",1.92],
                            ["C","N",1.57],
                            ["C","O",1.57],
                            ["N","H",1.11],
                            ["N","O",1.56],
                            ["N","N",1.55],
                            ["N","S",2.06], 
                            ["O","H",1.07],
                            ["O","S",1.52],
                            ["O","O",1.58], 
                            ["S","H",1.45],
                            ["S","S",2.17],
                            ["H","H",0.84]]
    
    etypes = [x.etype for x in atoms]
    datas = [x.data for x in atoms]
    target_etype = atom.etype
    target_data = atom.data
    bonds=[]
    for x in atoms:
        etype = x.etype
        data = x.data
        for item in maximal_bond_lengths:
            if (etype ==item[0] and target_etype == item[1]) or (etype ==item[1] and target_etype == item[0]):
                dist = math.dist(data,target_data)
                
                if dist<item[2]:
                    startpoint = target_data
                    endpoint = data
                    midpoint = (target_data+data)/2
                    start_type = target_etype
                    end_type = etype
                    
                    half_bond = Line(startpoint,midpoint,start_type,end_type)
                    bonds.append(half_bond)
                    
    return bonds

def get_bonded_atoms(atoms):
    bonded_atoms = []
    for atom in atoms:
        bonds = get_bond(atoms,atom)
        bonded_atom = bonded_Atom(bonds,atom)
        bonded_atoms.append(bonded_atom)
    return bonded_atoms




def get_protein_struct_from_pdb(pdbid, pdb_path, csv_path):

    pid = pdbid.strip()

    # ---- read partners from CSV ----
    partner1_ids = partner2_ids = None
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_pid = str(row.get("id", row.get("ID", ""))).strip()
            if row_pid != pid:
                continue
            partner1_ids = str(row["partner1"]).strip()
            partner2_ids = str(row["partner2"]).strip()
            break

    # --- parse PDB and extract chains ---
    parser = PDB.PDBParser()
    structure = parser.get_structure(pid, pdb_path)
    model = next(structure.get_models())  # Use first model
    chain_by_id = {c.id: c for c in model}
    
    def is_amino_acid_chain(chain):
        return any(is_aa(res, standard=True) for res in chain)

    partner1_ids = [cid for cid in partner1_ids if cid in chain_by_id and is_amino_acid_chain(chain_by_id[cid])]
    partner2_ids = [cid for cid in partner2_ids if cid in chain_by_id and is_amino_acid_chain(chain_by_id[cid])]

    missing = [c for c in partner1_ids + partner2_ids if c not in chain_by_id]
    if missing:
        raise ValueError(f"PDB {pid}: partner1 or partner2 has no amino-acid chain after filtering.")
        
    print(f"PDB {pid}: partner1:{partner1_ids} / partner2:{partner2_ids}")

    def build_one_chain_struct(chain):
        chain_struct, pre_struc = [], None
        for residue in chain:
            if residue.id[0] != " ":
                continue
            atoms = [Atom(a.get_coord(), a.element, a.fullname) for a in residue]
            cur_struc = get_bonded_atoms(atoms)

            if pre_struc is not None:
                for cur in cur_struc:
                    if getattr(cur.atom, "eid", "") == " N  ":
                        for pre in pre_struc:
                            if getattr(pre.atom, "eid", "") == " C  ":
                                start, end = pre.atom.data, cur.atom.data
                                mid = (start + end) / 2
                                pre.add_line(Line(start, mid, pre.atom.etype, cur.atom.etype))
                                cur.add_line(Line(end, mid, cur.atom.etype, pre.atom.etype))

            chain_struct += cur_struc
            pre_struc = cur_struc
        return chain_struct

    def build_group_struct(chain_ids):
        struct = []
        for cid in chain_ids:
            struct += build_one_chain_struct(chain_by_id[cid])
        return struct

    struct1 = build_group_struct(partner1_ids)
    struct2 = build_group_struct(partner2_ids)

    return struct1, struct2


# def get_ligand_struct_from_mol2(mol2file):
#     atoms,bondlist = get_atom_and_bond_list_from_mol2(mol2file)
#     bonded_atoms=[]
#     for atom in atoms:
#         bonded_atom=bonded_Atom([],atom)
#         bonded_atoms.append(bonded_atom)
        
#     for item in bondlist:
#         index1,index2 = item[0],item[1]
#         startpoint = bonded_atoms[index1-1].atom.data
#         endpoint = bonded_atoms[index2-1].atom.data
#         start_type = bonded_atoms[index1-1].atom.etype
#         end_type = bonded_atoms[index2-1].atom.etype
        
#         midpoint = (startpoint+endpoint)/2
#         line1 = Line(startpoint,midpoint,start_type,end_type)
#         line2 = Line(endpoint,midpoint,end_type,start_type)
#         bonded_atoms[index1-1].add_line(line1)
#         bonded_atoms[index2-1].add_line(line2)
#     return bonded_atoms

#########




class Gli_calculator():
    def __init__(self):
        self.struct1=[]
        self.struct2=[]
        self.result = {}
    
    def add_pair(self,bonded_atom1,bonded_atom2):
        self.struct1.append(bonded_atom1)
        self.struct2.append(bonded_atom2)
    
    def calculate_result(self):
        for bonded_atom1 in self.struct1:
            for bonded_atom2 in self.struct2:
                L = []
                for line1 in bonded_atom1.lines:
                    for line2 in bonded_atom2.lines:
                        gli = Gauss_linking_integral(line1, line2)
                        L.append(np.abs(gli)) #average crossing number, avoid cancelation of topo info.
                self.result[bonded_atom1,bonded_atom2] = L
                self.result[bonded_atom2,bonded_atom1] = L
        
    def get_result(self,bonded_atom1,bonded_atom2):
        if (bonded_atom1,bonded_atom2) in self.result:
            return self.result[bonded_atom1,bonded_atom2]
        else:
            return 0
    

####cutoff
def cutoff_struct(protein_struct,query_bonded_atom,cutoff=16):
    new_struct=[]
    for bonded_atom1 in protein_struct:
        atom1_pos = bonded_atom1.atom.data
        atom2_pos = query_bonded_atom.atom.data
        if math.dist(atom1_pos,atom2_pos)<cutoff:
            new_struct.append(bonded_atom1)
            break
    return new_struct


def get_mGLI(e1,e2,r1,r2,protein_struct1,protein_struct2,calculator):
    Gli_sum = []
    for bonded_atom1 in protein_struct1:
        if bonded_atom1.atom.etype == e1:
            gli_sum = 0
            for bonded_atom2 in protein_struct2:
                if bonded_atom2.atom.etype == e2:
                    dist = math.dist(bonded_atom1.atom.data,bonded_atom2.atom.data)
                    if dist >r1 and dist<r2 :
                        gli_sum+= calculator.get_result(bonded_atom1,bonded_atom2)
            Gli_sum.append(gli_sum)
    if Gli_sum == []:
        mGLI_feature = [0,0,0,0,0]
    else:
        Gli_sums =np.array(Gli_sum)
        mGLI_feature =[
            np.sum(Gli_sums),
            np.min(Gli_sums),
            np.max(Gli_sums),
            np.mean(Gli_sums),
            np.median(Gli_sums)
            ]  
    return mGLI_feature   



def get_inteval(n=10,minimal=2,maximal=11):
    scales = np.linspace(minimal,maximal,n)
    intevals=[]
    for i in range(n-1):
        r1 = scales[i]
        r2 = scales[i+1]
        intevals.append([r1,r2])
    return intevals

#######


def generate_mGLI_feature(pdbid):
    pdb_path = '/home/op98/protein_design/dataset/PP'+pdbid+'.ent.pdb'
    csv_path = '/home/op98/protein_design/dataset/PP/PP_anti/filtered_file1.csv'
    protein_struct1,protein_struct2= get_protein_struct_from_pdb(pdbid, pdb_path,csv_path)  
    
    print("struct,done")
    calculator = Gli_calculator()
    
    for query_bonded_atom in protein_struct2:
        new_struct = cutoff_struct(protein_struct1, query_bonded_atom)
        for bonded_atom1 in new_struct:
            calculator.add_pair(bonded_atom1,query_bonded_atom)
    calculator.struct1 = list(set(calculator.struct1))
    print("calculator,ready")
    print(len(calculator.struct1),len(calculator.struct2))
    calculator.calculate_result()
    print("calculation,done")
    el = ['C','N','O','S']#elements in be considered

    mGLI_feature=[]

    for e1 in el:
        for e2 in el:
            for scale in get_inteval():
                r1,r2 = scale
                mGLI_feature_1 = get_mGLI(e1,e2,r1,r2,protein_struct1,protein_struct2,calculator)
                mGLI_feature_2 = get_mGLI(e2,e1,r1,r2,protein_struct2,protein_struct1,calculator)
                mGLI_feature +=mGLI_feature_1
                mGLI_feature +=mGLI_feature_2
    return mGLI_feature,calculator


####
pdbid_list=['1j7v']


import time
Times=[]
for pdbid in pdbid_list:
    start =time.time()
    mGLI_feature,calculator = generate_mGLI_feature(pdbid)                    
    end = time.time()
    print(end-start)
    Times.append(end-start)