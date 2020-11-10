#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 12:34:00 2019

@author: elimiller
"""
import numpy as np
from numpy import pi
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Import matlab datafile
mat_contents = sio.loadmat('Testdata.mat')
data = mat_contents['Undata']

PLOTPATH = True
SAVEPATH = False

L = 15
n = 64

#Create spacial grid in a n,n,n space
x = np.linspace(-L, L, n+1)[0:n]
y = x
z = x
X, Y, Z = np.meshgrid(x, y, z)

#Create frequency space noramlized to 2pi
k = pi/L * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
#fftshift frequency domain
ks = np.fft.fftshift(k)
kx, ky, kz,  = np.meshgrid(ks, ks, ks)
average = np.zeros((n, n, n))


for j in range(len(data)):
    #Take a time slice of a (n,n,n) spacial data
    u = np.reshape(data[j, :],(n, n, n))
    #3d fft
    u_transform = np.fft.fftn(u)
    #fftshift of freqnecny data
    u_transform_shift = np.fft.fftshift(u_transform)
    #add each component to average out white noise
    average = average + u_transform_shift
#normalize the averaged data
normalized_average = np.abs(average / np.max(np.abs(average)))
#Find max of average data to have center of filter
max_index = np.unravel_index(
        np.argmax(normalized_average),(n,n,n), order='F')
#make a gaussian filter
k0 = [kx[max_index], ky[max_index], kz[max_index]]

def gauss_filter(noisy_data, index, alpha=.2, inputshifted=True):
    kxmax = kx[index]
    kymax = ky[index]
    kzmax = kz[index]
    filt = np.exp(-alpha*((kx-kxmax)**2 + (ky-kymax)**2 + (kz-kzmax)**2))
    
    if inputshifted:
        noisy_data = np.fft.fftshift(noisy_data)
    ReturnThis = noisy_data * filt
    return ReturnThis

#initalize data storage arrays before loop    
pt_index = np.zeros((20,3))
pt_location = np.zeros((20,3))
filt_spacial = np.zeros_like(data)

#This loop does the same as above, with added returning to spatial domain
#We could probably eliminate with shaping as (20, 60, 60, 60)
for j in range(len(data)):    
    spacial = np.reshape(data[j, :],(n, n, n))
    spacial_transform = np.fft.fftn(spacial)
    filtered_data = gauss_filter(spacial_transform, max_index)
    filtered_spacial = np.fft.ifftn(filtered_data)
    filt_spacial[j,:] = filtered_spacial.reshape(1, n**3)
    index = np.unravel_index(
            np.argmax(np.abs(filtered_spacial)),(n,n,n), order='F')
    pt_index[j,:] = index
    pt_location[j,:] = [X[index], Y[index], Z[index]] 
 #Important to pull location from meshgridded X, Y, Z, not the x, y, z

#Get the locaiton at time t=20
final_location  = pt_location[-1,:]
print('the marble\'s final location is' + str(final_location))

#Plot output
if PLOTPATH:
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pt_location[:,0], pt_location[:,1], pt_location[:,2])
    
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    
    fig2 = plt.figure(2)
    label = [['x', 'y'],['y','z'],['z', 'x']]
    for i in range(3):
        
        ax2 = plt.subplot(3,1,i+1)
        if i <= 1:
            ax2.plot(pt_location[:,i], pt_location[:, i+1], c='C0')
        else:
            ax2.plot(pt_location[:,i], pt_location[:, 0], c='C0')
        ax2.set_xlabel(label[i][0])
        ax2.set_ylabel(label[i][1])
        plt.tight_layout()
        
    if SAVEPATH:
        fig.savefig('marble_path.pdf')
        fig2.savefig('marble_traces.pdf')