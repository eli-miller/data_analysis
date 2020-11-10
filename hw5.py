#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:42:35 2019

@author: elimiller
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import skvideo.io
plt.close('all')

def get_x(filename, num_frames, size=(216, 384)):
    num_frames = num_frames
    shape = size
    
    video = skvideo.io.vread(
            filename,
            num_frames=num_frames, 
            as_grey=True, )[:, :, :, 0]
    A = []
    
    for frame in video:
        frame = resize(frame, shape)
        A.append(list(frame.ravel(order='C')))
    A = np.array(A).T
    average = A.mean(axis=0, keepdims=True)
    average = np.zeros_like(average)
    A = A - average
    return A, average

def DMD(A, rank, threshold):
        
    rank = rank
    threshold = threshold
    dt = 1
    
    X1 = A[:, :-1]
    X2 = A[:, 1:]
    
    U2, S_vec2, V2 = np.linalg.svd(X1, full_matrices=False)
    U = U2[:,0:rank]
    S = np.diag(S_vec2)[0:rank, 0:rank]
    V = V2.conj().T[:,0:rank]
    X = U @ S @ V.T
    if True:
        plt.figure()
        plt.plot(S_vec2 / np.sum(np.diag((S_vec2))))
    
    A_tilde = np.linalg.lstsq(U.conj().T @ X2 @ V, S, rcond=None)[0]
    mu, W = np.linalg.eig(A_tilde)
    Phi = np.linalg.lstsq((X2 @ V).T, S, rcond=None)[0] @ W
    
    omega = np.log(mu, dtype='complex128') / dt
    
    if True:
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Summary of Eigenvalues of $A_{tilde}$')
        axs[0].plot(omega.real, omega.imag,'o')
        axs[0].set_xlabel('Real Component')
        axs[0].set_ylabel('Imaginary Component')

        axs[1].plot(np.abs(omega),'o')
        axs[1].set_ylabel('Eigenvalue Magnitude')
        plt.tight_layout()
        
    omega_lr  = np.zeros_like(omega)
#    omega_sparse = np.zeros_like(omega)
    
    for j in range(len(omega)):
        #this could be faster with list comprehension
        if np.abs(omega[j]) <= threshold:
            omega_lr[j] = omega[j]

    t = np.arange(np.shape(A)[1])
    DMD_lr = np.zeros((rank, len(t)), dtype='complex128')
    b = np.linalg.lstsq(Phi, A[:,0], rcond=None)[0]
    
    for j in range(len(t)):
        DMD_lr[:, j] = b * np.exp(omega_lr * dt) 
    
    X_lr = Phi @ DMD_lr
    X_sparse = A - np.abs(X_lr)
    R = np.clip(X_sparse, a_min=None, a_max=0)
    X_lr = R + np.abs(X_lr)
    X_sparse = X_sparse - R
    X_recon = X_lr + X_sparse
        
    return X_lr, X_sparse, X_recon


def image_out(X_lr, X_sparse, A, average, frame, shape):
    og_image = np.abs(np.reshape((A+average)[:,frame], shape))
    lr_image = np.abs(np.reshape((X_lr+average)[:, frame], shape))
    sparse_image = np.abs(np.reshape((X_sparse+average)[:, frame], shape))
    sparse_image *= 255.0/sparse_image.max()
    recon_image = np.abs(np.reshape((A+average)[:, frame], shape))
    return [og_image, lr_image, sparse_image, recon_image]


def image(im_out, rank, savename):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.suptitle('Background Seperation: Rank=%d' %rank)
    axs[0, 0].imshow(im_out[0], cmap='gray')
    axs[0, 1].imshow(im_out[1], cmap='gray')
    axs[1, 0].imshow(im_out[2], cmap='gray')
    axs[1, 1].imshow(im_out[3], cmap='gray')
    
    titles = ['Original Frame', 
              'Low-Rank DMD (Background)', 
              'Sparse DMD (Foreground)',
              'Reconstructed Frame']
    
    axs[0, 0].set_title(titles[0])
    axs[0, 1].set_title(titles[1])
    axs[1, 0].set_title(titles[2])
    axs[1, 1].set_title(titles[3])
    
    for ax in axs.ravel(): ax.axis('off')
    plt.savefig(savename)
  
rank = 20
threshold = .05
filename = 'test3.mp4'
size = (216, 384)

A, average = get_x(filename, 100)
X_lr, X_sparse, X_recon = DMD(A, rank, threshold)

frame = 90
im_out= image_out(X_lr, X_sparse, A, average, frame, size)

image(im_out, rank, '%sfigure.pdf'%filename.replace('.mp4', ""))
