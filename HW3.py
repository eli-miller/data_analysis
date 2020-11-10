#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:29:54 2019

@author: elimiller
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.ndimage
import scipy.signal

def rgb2gray(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B 
    return np.array(gray)

def averageim(stack):
    if len(np.shape(stack)) == 4:
        stack = rgb2gray(stack)
    average_sum = np.sum(stack, axis=2) 
    return average_sum / np.ma.size(stack, axis=2)

def plot_fft(y):
    #frequencies normalized from [-1, 1]  for digital filter frequency
    y_freq = np.abs(np.fft.fftshift(np.fft.fft(y)))
    n = len(y_freq)
    k = 2/n * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
    #fftshift frequency domain
    ks = np.fft.fftshift(k)
    plt.figure()
    plt.plot(ks, y_freq)

def low_pass(y, pct_max_freq):
    b, a = scipy.signal.butter(1, pct_max_freq, 'lowpass' )
    y_filt = scipy.signal.lfilter(b, a, y)
    return y_filt

def get_com(
        framestack, 
        xbounds, 
        startval=0,  
        SHOW_AVERAGE=False, 
        FILTER_COM=True, 
        PLOT_TRACK=False):
    
    STORELEN = 226
    rightbnd = xbounds[0]
    leftbnd = xbounds[1]
    vid_frames = framestack[:,rightbnd:leftbnd,:,:] 
    average_frame = averageim(vid_frames)
    
    if SHOW_AVERAGE:
        plt.figure()
        plt.imshow(average_frame)
            
    com_x = []
    com_y = []
    for j in range(np.ma.size(vid_frames, axis=3)):
        frame = vid_frames[:,:,:,j]
        frame_bw = rgb2gray(frame)
        frame_delta =  frame_bw - average_frame
        #once we do this, we have transformed out of [0, 255] 
        #now our pixel values are relative!
        
        frame_delta[frame_delta < 0] = 0
        # send negative values of frame_delta to zero
        
        frame_filter = scipy.ndimage.gaussian_filter(frame_delta, 2)
        com = scipy.ndimage.measurements.center_of_mass(frame_filter)
        com_x.append(com[1])
        com_y.append(com[0])
        
        if PLOT_TRACK and j == 1:
            #this is solely to produce a nice figure for writeup
            bigfig, axs = plt.subplots(1, 4)
            axs[0].imshow(frame)
            axs[1].imshow(frame_bw)
            axs[2].imshow(frame_delta)
            axs[3].imshow(frame_filter)
            axs[3].plot(com[1], com[0], 'rx')

    #truncate to the same length
    com_x = com_x[startval:startval+STORELEN]
    com_y = com_y[startval:startval+STORELEN]
    
    if FILTER_COM:
        com_x_filter = low_pass(np.array(com_x), .15)
        com_y_filter = low_pass(np.array(com_y), .15)
        return com_x_filter, com_y_filter
    else:   
        return np.array(com_x), np.array(com_y)

def check_com(framestack, com_x, com_y, num_frames):
    for j in range(num_frames):
        plt.imshow(rgb2gray(framestack[:,:,:,j]))
        plt.plot(com_x[j], com_y[j], 'rx')
        plt.pause(.1)

def rank_approx(A_rel, rank):
    #this expects rows with mean 0
    U, S_vec, V = np.linalg.svd(A_rel)
    S = np.diag(S_vec)
    A_approx = U[:,0:rank] @ (S[0:rank, 0:rank] @ V[0:rank, :])
    return A_approx

def rank_plot(A, rank, coordinate):
    A_relative = A - A.mean(axis=1, keepdims=True)
    A_approx = rank_approx(A_relative, rank)
    plt.figure()
    plt.plot(A_relative[coordinate,:], '.-')
    plt.plot(A_approx[coordinate,:], '-')
    plt.title('Rank %d Reconstruction' %rank )
    plt.legend(['Original Data', 'Low-Rank Reconstruction'])

plt.close('all')  
cameras = [1, 2, 3]
examples = [1, 2, 3, 4]
bounds = [(250, 400),(200, 400),(100, 300)]
#indicies of windows of where to look for can
startvalstore = [[0, 10, 0],
                 [0, 10, 0], 
                 [0, 0, 0],
                 [0, 8, 0]]
#indicies of where to start videos to align them in time

for example in examples:
    bnd = 0
    STORELEN = 226
    startvals = startvalstore[bnd]
    A_temp = np.zeros((1, STORELEN))
    for camera in cameras:
        path = '/Users/elimiller/Desktop/AMATH482/HW3/camfiles/cam%d_%d.mat'%(
                camera, example)
        vid_frames_dict = loadmat(
                path)
        leftbnd = bounds[bnd][0]
        rightbnd = bounds[bnd][1]
        startval = startvals[bnd]
        
        vid_frames = vid_frames_dict['vidFrames%d_%d'%(camera, example)]
        
        if camera == 3:
            #the 3rd camera can be rotated because its sideways
            #we could forgo this and the SVD wouldn't care
            #however, it makes implementation easier for trimming frames
            vid_frames = np.rot90(vid_frames,k=-1)   
        
        com_x, com_y = get_com(
                vid_frames, (leftbnd, rightbnd), startval=startval, 
                SHOW_AVERAGE=False, PLOT_TRACK=True, FILTER_COM=True)        

        temp = np.vstack([com_x, com_y])
        A_temp = np.vstack([A_temp, temp])
    
        print('Example %d, Camera %d'%(
                example, camera))    
        print('Range of X=%.3f and Range of Y=%.3f' %(
                np.ptp(com_x), np.ptp(com_y)))
        
        bnd += 1
        
        if False:
            #Plots x and y coordinate of current camera
            fig, axs = plt.subplots(2,1, sharey=True)
            axs[0].plot(com_x, '.-')
            axs[0].set_title('x coordinate')
            axs[1].plot(com_y, '.-')
            axs[1].set_title('y coordinate')

    #Store in different matricies so that we don't have to run this again
    if example == 1:
        A1 = A_temp[1::]
    if example == 2:
        A2 = A_temp[1::]
    if example == 3:
        A3 = A_temp[1::]
    if example == 4:
        A4 = A_temp[1::]
        

#Plot all components for each case.  Used to align video frames by hand
for A in [A1, A2, A3, A4]:
    plt.figure()
    for j in range(6): plt.plot(A[j, :])
    plt.legend(['x1', 'y1', 'x2', 'y2', 'x3', 'y3'], loc='lower right')   
    
plt.figure()
#plot of Singluar values for each case
for A in [A1, A2, A3, A4]:
    A_rel = A - A.mean(axis=1, keepdims=True)
    U, S, V = np.linalg.svd(A_rel)
    plt.plot(range(1, 6+1), S / np.sum(S), 'o-')
    plt.yscale('linear')
    plt.ylabel('% Energy Captured')
    plt.xlabel('Mode')
    plt.legend(('Example 1 (1D)',
             'Example 2 (1D Shake)',
             'Example 3 (2D)', 
             'Example 4 (3D)' ))
    
    plt.title('Singular Values (Normalized)')
    plt.savefig('SingularValues.pdf')


#Produce plots of rank reconstruction
#Investigate mean squared error for each case 

#plt.close('all')
error_store = np.zeros((6, 6, 4))

rankval = np.arange(1, 6+1)
coordinates = np.arange(0, 5+1)

As = [A1, A2, A3, A4]
#As = [A4]
layer = 0
for A in As:
    for rank in rankval:
        for j in range(len(coordinates)):
            coordinate = coordinates[j]            
            A_relative = A - A.mean(axis=1, keepdims=True)
            A_approx = rank_approx(A_relative, rank)
            error = np.sqrt(np.sum((A_approx[j,:] - A_relative[j,:])**2))
            error_store[coordinate, rank-1, layer] = error
    layer += 1

if False:
    plt.plot(A_relative[coordinate,:], '.-')
    plt.plot(A_approx[coordinate,:], '-')
    plt.title('Rank %d Reconstruction with error %d' %(rank, error))
    plt.legend(['Original Data', 'Low-Rank Reconstruction'])
    

#Lets make an informative rank reconstruction figure!
#plt.close('all')
trimstart = 26
#trim off the beginning to condense in the y direction

A_relative = (A1 - A1.mean(axis=1, keepdims=True))[:,trimstart::]
rankval = [1, 2]
ax_index = [(0, 0), (0, 1), (1, 0), (1, 1), (2,0), (2,1)]

fig, axs = plt.subplots(3, 2, sharex=True, sharey=False)
for j in range(6):
    axs[ax_index[j]].plot(A_relative[j,:], '.-', linewidth=1, markersize=4)
    for rank in rankval:
        A_approx = rank_approx(A_relative, rank)
        axs[ax_index[j]].plot(A_approx[j,:])

fig.suptitle('Rank Reconstructions: Case 1')
fig.legend(['Original', 'Rank 1', 'Rank 2'])

axs[0,0].set_title('X Coordinate')
axs[0,1].set_title('Y Coordinate')
for j in range(3):
    axs[(j,0)].set_ylabel('Camera %d' %(j+1))
    
plt.savefig('RankReconstruction.pdf')


