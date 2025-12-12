#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:41:52 2023

@author: shenli
"""

import numpy as np
from GLI_Functions import GLI_feature_standard,Gauss_linking_integral,line_classification,get_protein_ca_atom_coordinate,GLI_feature
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix


def read_ca_coordinates(filename):
    coords = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[13:15] == "CA":  # Cα only
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])

                    resname = line[17:20].strip()
                    chain = line[21].strip()
                    resseq = line[22:26].strip()
                except ValueError:
                    continue
    return np.array(coords)

def line_ca(cloudpoints,index):
    lines=[]
    n = cloudpoints.shape[0]-1   
    if index == 0:
        end2 = (cloudpoints[0]+cloudpoints[1])/2
        end1 = cloudpoints[0]
        lines.append(((end1,end2)))
    
    elif index == n:
        end1 = (cloudpoints[n-1]+cloudpoints[n])/2
        end2 = cloudpoints[n]
        lines.append((end1,end2))
    else:
        lines.append(((cloudpoints[index-1]+cloudpoints[index])/2,cloudpoints[index]))
        lines.append((cloudpoints[index],(cloudpoints[index]+cloudpoints[index+1])/2))
        
    return lines
        
def GLIMat(cloudpoints):
    n = cloudpoints.shape[0]
    M = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i+1,n):
            lines1 = line_ca(cloudpoints,i)
            lines2 = line_ca(cloudpoints,j)
            L = 0
            T = 0
            for line1 in lines1:
                for line2 in lines2:
                    #print(line1,line2)
                    L += np.abs(Gauss_linking_integral(line1,line2))
                    T += Gauss_linking_integral(line1,line2)
            
            K = np.abs(T)

            
            
            M[i,j]+= L
            M[j,i]+= L
    return M





#datapath = './365'
#pdbid = '2GZQ'
#filepath = datapath+'/'+pdbid+'_CA_A2.pdb'

pdbid ='1a22'
filepath = '/home/op98/protein_design/dataset/PP/PP2/'+pdbid+'.ent.pdb'
CA_coor = read_ca_coordinates(filepath)
#CA_coor, labels = get_protein_ca_atom_coordinate(pdbid,filepath)

M = GLIMat(CA_coor)
Dist = distance_matrix(CA_coor,CA_coor)


def TPM_exp(sigma,k,distM):
    TPMexpM = np.zeros_like(distM)
    n,m = distM.shape
    for i in range(n):
        for j in range(m):
            if i != j:
                TPMexpM[i,j] += np.exp(-(distM[i,j]/sigma)**k)
            if i == j :
                TPMexpM[i,j] += - distM[i,j]
    return TPMexpM

def filt(matrix,num):
    minimal = np.min(matrix)
    maximal = np.max(matrix)
    seg = (maximal-minimal)/num
    matrix_list=[]
    
    for i in range(num):
        cur_matrix = np.where(np.logical_and(minimal+i*seg<=matrix,matrix<=minimal+(i+1)*seg),1,0)
        matrix_list.append(cur_matrix)
    return matrix_list


TPMexpM = TPM_exp(10,2,Dist)

matrix_list = filt(TPMexpM,4)
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

N = (M-M.mean())/M.std()
#N = np.flip(N,axis=0)
plt.imshow(N, cmap='rainbow', interpolation='nearest',vmin=0,vmax=1)
plt.xticks([])
plt.yticks([])
#plt.colorbar()
xlims = plt.xlim()
ylims = plt.ylim()

# plt.hlines(ylims[1],xmin=18,xmax=37,color='y', lw=1, zorder=4, clip_on=False,linewidth=3)
# plt.vlines(xlims[0],ymin=18,ymax=37,color='y', lw=1, zorder=4, clip_on=False,linewidth=3)

# plt.hlines(ylims[1],xmin=63,xmax=81,color='y', lw=1, zorder=4, clip_on=False,linewidth=3)
# plt.vlines(xlims[0],ymin=63,ymax=81,color='y', lw=1, zorder=4, clip_on=False,linewidth=3)


# plt.hlines(ylims[1],xmin=40,xmax=45,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)
# plt.vlines(xlims[0],ymin=40,ymax=45,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)

# plt.hlines(ylims[1],xmin=52,xmax=61,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)
# plt.vlines(xlims[0],ymin=52,ymax=61,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)

# plt.hlines(ylims[1],xmin=2,xmax=13,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)
# plt.vlines(xlims[0],ymin=2,ymax=13,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)

# plt.hlines(ylims[1],xmin=85,xmax=97,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)
# plt.vlines(xlims[0],ymin=85,ymax=97,color='g', lw=1, zorder=4, clip_on=False,linewidth=3)
plt.savefig(f'gli_{pdbid}.pdf',dpi=300,bbox_inches='tight',format='pdf')
plt.show()

for dist in matrix_list:
    filtered_M  = M*dist
    
    N = (filtered_M-filtered_M.mean())/filtered_M.std()
    #N = np.flip(N,axis=0)
    plt.imshow(N, cmap='rainbow', interpolation='nearest',vmin=0,vmax=1)
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    plt.show()