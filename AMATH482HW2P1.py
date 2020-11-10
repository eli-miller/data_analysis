#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:29:53 2019

@author: elimiller
"""
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import scipy.io
plt.ioff()
#plt.ion()
#%%

def fft2flip(args):
        ReturnThis = np.abs(np.fft.fftshift(np.fft.fft(args)))
        return ReturnThis
    
def gaussian(unfiltered, t, location, width=.2):
    #scipy.signal.gaussian
    sigma = width
    coeff = (1/(sigma*np.sqrt(2* pi)))
    denom = 2 * sigma**2
    gauss_filter = coeff * np.exp(-(t-location)**2 / denom)
    return gauss_filter * unfiltered

def mexihat(unfiltered, t, location, width=.2):
    sigma = width
    t_instant = t - location
    psi = 2/(np.sqrt(3*sigma)*pi**(1/4)) * (
            1 - (t_instant)/(sigma)**2) * np.exp(
                    -(t_instant**2)/(2*sigma**2))
    return psi * unfiltered

def step(unfiltered, t, location, width=.2):
    bool_array = np.abs(t-location) < width/2
    stepfilt = np.array(bool_array * 1, dtype='float64')
    return stepfilt * unfiltered

def spectrogram(
        v, fs, width, samplemod=1, 
        window='gaussian', makefig=False, numpts='100'):
   
    window_dict = {'gaussian': gaussian, 'mexihat': mexihat, 'step': step}
    L = len(v)/fs
    t = np.arange(0, len(v))/fs
    tshift = np.linspace(0, L, numpts)
    n = len(v)
    k = pi/L * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
    ks = np.fft.fftshift(k)/(2*pi)
    gabor = np.zeros((len(t),len(tshift)))
    
    for j in range(len(tshift)):
        v_gabor = window_dict[str(window)](v, t, tshift[j], width=width)
        v_gabor_trans_shift = fft2flip(v_gabor)
        gabor[:,j] = v_gabor_trans_shift
        if makefig:
            #make figure for looking at how gabor works
            if j == 20:
                fig, axs = plt.subplots(4,1)
                titles = [
                        'Original Signal and Filter',
                        'Filtered Signal',
                        'FFT of Original Signal',
                        'FFT of Filtered Signal']
                axs[0].plot(t, v)
                axs[0].plot(
                        t, gaussian(np.ones_like(v), t, tshift[j])/np.amax(
                        gaussian(np.ones_like(v), t, tshift[j])
                        )
        )
                axs[1].plot(t, v * gaussian(v, t, tshift[j]))
                axs[2].plot(ks, fft2flip(v))
                axs[3].plot(ks, fft2flip(v*gaussian(v, t, tshift[j])))
                
#                plt.suptitle('Visualization of Gabor Transform')
                
                count = int(0)
                for title in titles:
                    axs[count].set_title(title)
                    count += 1
                plt.tight_layout()
    return gabor 
#%%
load_dict = scipy.io.loadmat('handel')
v = np.reshape(load_dict['y']/2, len(load_dict['y']))
v = v[:-1]
#k = pi/L * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
#ks = np.fft.fftshift(k)
fs = 8192
L = len(v)/fs
n = len(v)
k = pi/L * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
ks = np.fft.fftshift(k)

ratio = np.array([1, .5, .1, .01])
widthvals = len(v)/fs * ratio

#%%
# Filter Width

for width in widthvals:
    gabor_width = spectrogram(
            v, fs, width, samplemod=1.5, window='gaussian')
    fig = plt.figure()
    plt.pcolormesh(gabor_width)
    plt.xlabel('Time (s)')
    plt.xticks([0, 50, 100], [0, 4.46, 8.92])
    plt.ylabel('Frequency($\omega$)')
    plt.yticks([0, len(v)/2, len(v)],[] )
    plt.title('Width = %.3f' %width)
    pathname = '%.3fwidth' %width
    pathname = pathname.replace('.', 'point')
    plt.savefig(pathname)
    plt.close(fig)
#%%
#tsample = [10, 100, 1000]
tsample = [1000]
width = widthvals[-1]

for numpts in tsample:
    gabor_sample = spectrogram(
            v, fs, width, samplemod=1.5, window='gaussian', numpts=numpts)
    fig = plt.figure()
    plt.pcolormesh(gabor_sample)
    plt.xlabel('Time (s)')
    plt.xticks([0, numpts/2, numpts], [0, 4.46, 8.92])
    plt.ylabel('Frequency($\omega$)')
    plt.yticks([0, len(v)/2, len(v)],[] )
    plt.title('%d sample points' %numpts)
    plt.savefig('%d points' %numpts)
    plt.clf()
#    plt.show()

#%%
#Filter Type
width = widthvals[-1]
numpts=100
gabor_gauss = spectrogram(
        v, fs, width, samplemod=1.5, window='gaussian')
gabor_mexihat = spectrogram(
        v, fs, width, samplemod=1.5, window='mexihat')
gabor_step = spectrogram(
        v, fs, width, samplemod=1.5, window='step')

fig, axs =  plt.subplots(3, 1, sharex=True, sharey=True)
titles = ['Gaussian', 'Mexihat', 'Step']
cmap = 'viridis'
if True:
    axs[0].pcolormesh(gabor_gauss, cmap=cmap)
    axs[0].set_title(titles[0])
    axs[1].pcolormesh(gabor_mexihat, cmap=cmap)
    axs[1].set_title(titles[1])
    axs[2].pcolormesh(gabor_step, cmap=cmap)
    axs[2].set_title(titles[2])
    plt.setp(axs, 
             xticks=[0, numpts/2, numpts], 
             xticklabels=[0, 4.46, 8.92], 
             yticklabels=[])
    plt.tight_layout()
    plt.show()
    plt.savefig('window')
    
    
    
    
    
    
    
