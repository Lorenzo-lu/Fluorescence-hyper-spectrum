# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:49:15 2020

@author: yluea
"""

import numpy as np;
#from skimage.transform import rescale, resize, downscale_local_mean;
#from skimage.transform import resize;

"""
Directly import Haar_2d

"""

def Resize(mat,new_size):
    r,c = new_size;
    #return resize(mat, (r,c), anti_aliasing=False);
    r0,c0 = mat.shape;
    mat2 = np.zeros((r,c));
    for i in range(r):
        y = int((i) * (r0) / (r));
        for j in range(c):
            x = int((j) * (c0) / (c));
            mat2[i,j] = mat[x,y];
    
    return mat2;
    
    

def Haar_tree(seed_raw):
    r,c = seed_raw.shape;
    r2 = r//2;
    c2 = c//2;
    seed = Resize(seed_raw, (r2,c2));
    
    k1 = np.array([[1,0],[0,0]]);
    k2 = np.array([[0,1],[0,0]]);
    k3 = np.array([[0,0],[1,0]]);
    k4 = np.array([[0,0],[0,1]]);
    
    out = np.zeros((2*r,2*c));
    out[:r,:c] = np.kron(k1,seed);
    out[:r,c:] = np.kron(k2,seed);
    out[r:,:c] = np.kron(k3,seed);
    out[r:,c:] = np.kron(k4,seed);
    
    return out;

def make_tensor(n, seed,edge):
    """
    Make a tensor consisting of masks; each layer along axis 2 is a independent
    mask.
    """
    r = 2**n;
    tensor = np.zeros((edge,edge, 2**(2*n)));
    layer = 0;## index along z axis
    
    sub = np.zeros((r,r));
    for i in range(r):
        for j in range(r):
            sub[i,j] = 1;
            tensor[:,:,layer] = Resize(np.kron(sub, seed),(edge,edge));
            sub[i,j] = 0;
            layer += 1;
    
    return tensor;

def Haar_2d(edge):
    """
    Return a Haar basis matrix of 2D images
    """
    
    if np.log2(edge) != int(np.log2(edge)):
        print("Please have the edge with length of 2^n");
        return;
        
    #n = int(np.log2(edge));
    
    haar_tensor = np.zeros((edge, edge, edge**2));
    
    seed1 = np.array([[1,1],[1,1]]);
    seed2 = np.array([[1,-1],[1,-1]]);
    seed3 = np.array([[1,1],[-1,-1]]);
    seed4 = np.array([[1,-1],[-1,1]]);
    
    kernel1 = Resize(seed1, (edge, edge));
    kernel2 = Resize(seed2, (edge, edge));
    kernel3 = Resize(seed3, (edge, edge));
    kernel4 = Resize(seed4, (edge, edge));
    haar_tensor[:,:,0] = kernel1;
    haar_tensor[:,:,1] = kernel2;
    haar_tensor[:,:,2] = kernel3;
    haar_tensor[:,:,3] = kernel4;
    
    layer = 4;
    kernel_edge = edge;
    N = 1;
    
    while kernel_edge >= 4:
        t1 = make_tensor(N, seed2, edge);
        L = len(t1[0,0,:]);
        haar_tensor[:,:,layer:layer+L] = t1;
        layer += L;
        
        t2 = make_tensor(N, seed3, edge);
        L = len(t2[0,0,:]);
        haar_tensor[:,:,layer:layer+L] = t2;
        layer += L;
        
        t3 = make_tensor(N, seed4, edge);
        L = len(t3[0,0,:]);
        haar_tensor[:,:,layer:layer+L] = t3;
        layer += L;
        
        kernel_edge /= 2;
        N += 1;
        
    haar_2d_basis = np.zeros((edge**2, edge**2));
    for i in range(edge**2):
        vec = np.reshape(haar_tensor[:,:,i], (1,-1));
        abs_sum = np.sum(np.abs(vec));
        haar_2d_basis[i,:] = vec /(abs_sum**0.5);
        
    return haar_2d_basis;
        
        
    
    
    

