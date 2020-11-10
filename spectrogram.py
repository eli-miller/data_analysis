#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:29:37 2019

@author: elimiller
"""

import numpy as np
from numpy import pi


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

def spectrogram(v, fs, width, samplemod=1, window='gaussian'):
   
    window_dict = {'gaussian': gaussian, 'mexihat': mexihat, 'step': step}
    L = len(v)/fs
    t = np.arange(0, len(v))/fs
    tshift = np.linspace(0, L, L/width * samplemod)
    gabor = np.zeros((len(t),len(tshift)))
    
    for j in range(len(tshift)):
        v_gabor = window_dict[str(window)](v, t, tshift[j], width=width)
        v_gabor_trans_shift = fft2flip(v_gabor)
        gabor[:,j] = v_gabor_trans_shift
        
    return gabor 