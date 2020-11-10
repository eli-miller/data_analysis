#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 2 Part1
"""
import scipy
import scipy.io as sio
from scipy import signal
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
#%%
load_dict = scipy.io.loadmat('handel')
v = np.reshape(load_dict['y']/2, len(load_dict['y']))
v = v[:-1]
#remove last element of v to make even
fs = 8192 #this is from the handel file?
L = len(v)/fs #Check this is correct
n = len(v)
t = np.arange(0, len(v))/fs
k = pi/L * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
ks = np.fft.fftshift(k)


PLOT = True


def fft2flip(args):
    ReturnThis = np.abs(np.fft.fftshift(np.fft.fft(args)))
    return ReturnThis

def gaussian(unfiltered, t, location, width=.2):
    #scipy.signal.gaussian
    tau = 1/ (2*width)
    gauss_filter = np.exp(-tau*(t-location)**2)
    return gauss_filter * unfiltered

def mexihat(unfiltered, t, location, width=.2):
    tau = width
    t_instant = t - location
    psi = 2/(np.sqrt(3*tau)*pi**(1/4)) * (
            1 - (t_instant)/(tau)**2) * np.exp(
                    -(t_instant**2)/(2*tau**2))
    return psi * unfiltered

def step(unfiltered, t, location, width=.2):
    bool_array = np.abs(t-location) < width/2
    stepfilt = np.array(bool_array * 1, dtype='float64')
    return stepfilt * unfiltered


#sigma_t = 0
widthval = np.linspace(.01, .1, 5)
 #width = .2
#def spectrogram(signal, tshift, width, window):
#    if window == 'gaussian'
#     
w = 0
for width in widthval:
    
    tshift = np.linspace(0, L, )
    gabor_gauss = np.zeros((len(t),len(tshift)))
    gabor_mexih = np.zeros((len(t),len(tshift)))
    gabor_step = np.zeros((len(t),len(tshift)))
    
    for j in range(len(tshift)):
        
        v_gauss = gaussian(v, t, tshift[j], width=width)
        v_gauss_trans_shift = fft2flip(v_gauss)
        gabor_gauss[:,j] = v_gauss_trans_shift
    #    sigma_t += tshift[j]**2 * np.abs(v_gauss)**2
    
        v_mexih = mexihat(v, t, tshift[j], width=width)
        v_mexih_trans_shift = fft2flip(v_mexih) 
        gabor_mexih[:,j] = v_mexih_trans_shift
        
        v_step = step(v, t, tshift[j], width=width)
        v_step_trans_shift = fft2flip(v_step)
        gabor_step[:,j] = v_step_trans_shift
         
    if PLOT:   
#        plt.close('all')
        f, axarr = plt.subplots(3, 5, sharex=True)
        plt.suptitle('Filter Width = %f' %width)
    #    plt.add_subplot(311)
        axarr[0, w].pcolormesh(gabor_gauss, cmap='magma')
        axarr[0, w].set_title('Gaussian')
    #    plt.subplot(312)
        axarr[1, w].pcolormesh(gabor_mexih, cmap='magma')
        axarr[1, w].set_title('Mexican Hat')
    #    plt.subplot(313)
        axarr[2, w].pcolormesh(gabor_step, cmap='magma')
        axarr[2, w].set_title('Step Function')
        w += 1
#        plt.savefig('a "%f width".jpeg ' %width)

    #    ax1.plot(t, v, t, gaussian(np.ones_like(t), t, timestep ))
    #    ax2.plot(ks, np.abs(np.fft.fftshift(np.fft.fft(v))))
    #    ax3.plot(t, v_filt)  
    #    ax4.plot(ks, np.abs(v_filt_trans_shift))
    #    
    #    plt.pause(1)
    #    
