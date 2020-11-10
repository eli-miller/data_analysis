#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:32:03 2019

@author: elimiller
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
#%%

directory = '/Users/elimiller/Desktop/AMATH482/HW4/CroppedYale/'

filename = 'yaleB'
indicies = np.arange(1, 40)
indicies = np.delete(indicies, 13)
#indicies = np.arange(1, 3)

A1 = np.zeros((1, 32256))

for index in indicies:
    folder = filename + str(index).zfill(2)
    print(folder)
    folder_directory = directory + folder
    dirs = os.listdir(folder_directory)
    for file in dirs:
        path = directory + '/' + folder + '/' + file
        image = plt.imread(path, 'gif')
        long_image = np.reshape(image, (1, 32256))
        A1 = np.append(A1, long_image, axis=0)
A1 = A1[1::,:]

A2 = np.zeros((1, 77760))

uncropped_dir = '/Users/elimiller/Desktop/AMATH482/HW4/yalefaces_uncropped/yalefaces'
for file in os.listdir(uncropped_dir):
    image = plt.imread(uncropped_dir + '/' + file, 'gif')
    long_image = np.reshape(image, (1, 77760))
    A2 = np.append(A2, long_image, axis=0)
A2 = A2[1::,:]    
#%%
U1, S1_vec, V1 = np.linalg.svd(A1-A1.mean(axis=1, keepdims=True), full_matrices=False)
U2, S2_vec, V2 = np.linalg.svd(A2-A2.mean(axis=1, keepdims=True), full_matrices=False)
S1 = np.diag(S1_vec)
S2 = np.diag(S2_vec)
#%%
fig, axs = plt.subplots(2, 2, sharey='row')
length = len(S2_vec)
pct = .68

axs[0,0].plot(np.cumsum(S1_vec)/ np.sum(S1_vec), '-')
axs[0,0].set_title('Cropped')
axs[0,0].set_ylabel('Cumulative Energy')
#axs[0,0].set_xlabel('Modes')
axs[0,1].plot(np.cumsum(S2_vec)/ np.sum(S2_vec), '-')
axs[0,1].set_title('Uncropped')
#axs[0,1].set_ylabel('Cumulative Energy')
#axs[0,1].set_xlabel('Modes')

axs[1,0].plot((S1_vec)/ np.sum(S1_vec), '-')
#axs[1,0].set_title('Cropped Spectrum')
axs[1,0].set_ylabel('Normalized Energy')
axs[1,0].set_xlabel('Modes')
axs[1,1].plot((S2_vec)/ np.sum(S2_vec), '-')
#axs[1,1].set_title('Uncropped Spectrum')
#axs[1,1].set_ylabel('Normalized Energy')
axs[1,1].set_xlabel('Modes')

plt.tight_layout()

plt.savefig('SVD_Comparison.pdf')
#%%



def rank_face(SVD, rank, facenum, shape):
    U = SVD[0]
    S = SVD[1]
    V = SVD[2]
    low_rank = U[:,0:rank] @ S[0:rank, 0:rank] @ V[0:rank, :]
    image = np.reshape(low_rank[facenum,:], shape)
    return image


shapes = [(168, 192), (320, 243)]
rankvals = [10, 150, 166]
SVDstore = [[U1, S1, V1],[U2, S2, V2]]

fig, axs = plt.subplots(len(rankvals), len(SVDstore))
facenum = 0
x = 0
for SVD in SVDstore:
    y = 0
    shape = shapes[x]
    for rank in rankvals:
        
        image = rank_face(SVD, rank, 10, shape)
        axs[y, x].imshow(image, cmap = 'gray')
        print('x:%d,  y:%d'%(x,y))
        y += 1 
    x += 1
    #%%
rank = 50
low_rank = U2[:,0:rank] @ (np.diag(S2)[0:rank, 0:rank] @ V2[0:rank, :])
for j in np.arange(25):
    plt.imshow(np.reshape(low_rank[j,:], (243,320)), cmap='gray')
    plt.pause(.5)
    

A_trunc = A[0:100, 0:-1:2]
A_trunc = A_trunc - A_trunc.mean(axis=1, keepdims=True)

plt.plot(S_vec / np.sum(S_vec), 'o-')
plt.plot(np.cumsum(S / np.sum(S)))

#%%

rank = 20

S = np.diag(S_vec)
A_approx = U[:,0:rank] @ (S[0:rank, 0:rank] @ V[0:rank, :])


for j in np.arange(0, 50):
    plt.clr()
    test_shape = np.reshape(A_approx[j,:], (192, 168))
    plt.imshow(test_shape, cmap='gray')
    plt.pause(.3)




















