#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:29:53 2019

@author: elimiller
"""

#from spectrogram import spectrogram
from scipy.io import wavfile
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
plt.ion()
#%%

rate, music1, = wavfile.read(
        '/Users/elimiller/Desktop/AMATH482/HW2/music1.wav')
rate2, music2 = wavfile.read(
         '/Users/elimiller/Desktop/AMATH482/HW2/music2.wav')
tr_piano = 16
tr_rec = 14
t = np.linspace(0, tr_piano, len(music1))
rate_analytical1 = len(music1) / tr_piano
rate_analytical2  = len(music2 / tr_rec)

#k = 1/(L) * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
#ks = np.fft.fftshift(k)


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

def spectrogram(v, fs, width, samplemod=1, window='gaussian', numpts='100'):
   
    window_dict = {'gaussian': gaussian, 'mexihat': mexihat, 'step': step}
    L = len(v)/fs
    t = np.arange(0, len(v))/fs
    tshift = np.linspace(0, L, numpts)
    gabor = np.zeros((len(t),len(tshift)))
    
    for j in range(len(tshift)):
        v_gabor = window_dict[str(window)](v, t, tshift[j], width=width)
        v_gabor_trans_shift = fft2flip(v_gabor)
        gabor[:,j] = v_gabor_trans_shift
    return gabor 

def gaussian_freq(width):
    sigma = width
    coeff = (1/(sigma*np.sqrt(2* pi)))
    denom = 2 * sigma**2
    gauss_filter = coeff * np.exp(-(ks)**2 / denom)
    return gauss_filter / np.amax(gauss_filter)

#%%
    numpts = 400
width  = .1
gabor_out1 = spectrogram(
        music1, rate_analytical1, width, samplemod=1, window='mexihat', numpts=numpts) 
#piano
cmap = 'YlGnBu'

band = 200000
#lowbnd = int(len(music1)-band/2)
lowbnd = 354000
#highbnd = int(len(music1)+band/2)
highbnd = 356250
gabor_trim1 = gabor_out1[lowbnd:highbnd, :]
n = len(music1)
L = tr_piano
k = 1/(L) * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
ks = np.fft.fftshift(k)
kstrim1 = ks[lowbnd:highbnd]

#plt.pcolormesh(gabor_trim, norm=matplotlib.colors.LogNorm(vmin=gabor_trim.min(), vmax=gabor_trim.max()))
#plt.imshow(gabor_out, cmap='magma', aspect='auto', origin='lower')
if False:
    plt.subplot(211)
    plt.pcolormesh(gabor_trim1, cmap=cmap)
    note_index1 = [1835, 1300, 800]
    note_freq1 = []
    for notes in note_index1: note_freq1.append(kstrim1[notes].round(2))
    plt.title('Piano')
    plt.yticks(note_index1, note_freq1)
    plt.ylabel('Frequency (Hz)')
    plt.xticks(np.linspace(0, numpts, 5), np.linspace(0, L, 5))
    plt.xlabel('Time (s)')
    #plt.plot(tshift, 1350*np.ones_like(t))

#recorder
n = len(music2)
tr_rec = 14
L = tr_rec
width  = .01*L
rate_analytical2  = len(music2) / tr_rec

gabor_out2 = spectrogram(
        music2, rate_analytical2, width, samplemod=1, window='mexihat', numpts=numpts) 



k = 1/(L) * np.concatenate((np.arange(0, (n/2)), np.arange(-n/2, 0)))
ks = np.fft.fftshift(k)
lowbnd = 324500
highbnd = 330000

gabor_trim2 = gabor_out2[lowbnd:highbnd, :]
kstrim2 = ks[lowbnd:highbnd]

if False:
    plt.subplot(212)
    plt.title('Recorder')
    plt.pcolormesh(gabor_trim2, cmap=cmap)=
    note_index2 = [3800, 2100, 800]
    note_freq2 = []
    for notes in note_index2: note_freq2.append(kstrim2[notes].round(2))
    plt.yticks(note_index2, note_freq2)
    plt.ylabel('Frequency (Hz)')
    plt.xticks(np.linspace(0, numpts, 5), np.linspace(0, L, 5))
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('part2')