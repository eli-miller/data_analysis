#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:16:07 2019

@author: elimiller
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import librosa
import sklearn



#%%
def get_features(data, rate, samples, label, plot_spectrograms=False):
    
    data = np.array(data, dtype='float64')
    
    sample_length = 5 * rate
    #indiceis for 5 second clips
    num_samples = int(np.floor(len(data) / sample_length))
    #round so that we can reshape without residuals
    
    trunc_index = sample_length * num_samples
    
    data_mat = np.reshape(data[0:trunc_index], (num_samples, sample_length))
    np.random.shuffle(data_mat)
    
    if samples == 'max':
        samples = num_samples
    data_set = data_mat[0:samples, :]
    
    feature = [label] * np.shape(data_set)[0]
    #make a list of input genre for the album data
    for j in range(samples):
        f, t, Sxx =  scipy.signal.spectrogram(
                data_set[j,:], rate, scaling='spectrum', mode='magnitude')
        if plot_spectrograms:
            plt.figure()
            plt.pcolormesh(t, f, Sxx, cmap='plasma')
            plt.title(label.upper())
            plt.xlabel('Time $t$')
            plt.ylabel('Frequency $\omega$')
        
        if j ==0:
            spectrogram_data = np.zeros((samples, np.size(Sxx.flatten())))
        
        spectrogram_data[j, :] = Sxx.flatten()
    return spectrogram_data, feature

#%%
genres = np.array(['jazz', 'rock', 'edm'])
#genres = np.array(['rock'])
samplenums = np.array(['0', '1', '2'])
#samplenums = np.array(['0'])


MAKEFILES = True
SAVEFILES = False
LOADFILES = False


short_length = 60*25
#duration=short_length


#samples_per_file = int(short_length / 5)
samples_per_file = 'max'


A = []
labels = []



for genre in genres:
    for samplenum in samplenums:
        if MAKEFILES:
            path = '/Users/elimiller/Desktop/AMATH482/HW4/Audio_Files/' + genre + samplenum + '.mp3'
            data, rate = librosa.load(
                    path, res_type='kaiser_fast', offset=0, duration  = 25*60)
        if LOADFILES:
            rate = 22050
            data = np.load(genre+samplenum+'.npy')
            
        if SAVEFILES:
            np.save('%s%s'%(genre, samplenum),data)
            
        print('genre: %s, sample: %s loaded' %(genre, samplenum))
        
        x_data, y_data = get_features(
                data, 
                rate, 
                samples_per_file, 
                genre,
                plot_spectrograms=False)

        A.append(x_data)
        labels.append(y_data)
       
A_array = np.array(A)            
A_new = np.reshape(
        A_array, 
        (np.shape(A_array)[0]*np.shape(A_array)[1], np.shape(A_array)[2]))

    
labels_array = np.array(labels)
labels_new = np.ravel(np.reshape(labels_array, (np.size(labels_array),1)))


A_new_centered = A_new - A_new.mean(axis=1, keepdims=True)
U, S, V = np.linalg.svd(A_new_centered, full_matrices =False)
plt.figure()
plt.plot(S/np.sum(S), 'o-')
plt.title('Singular Value Spectrum of Audio Data')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#suppress warnings for printouts
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

rank = 25

representation = A_new_centered @ V.T[:, 0:rank]

x_train, x_test, y_train, y_test = train_test_split(
        representation, labels_new, train_size=.80)

svm_clf = LinearSVC()
svm_clf.fit(x_train, y_train)
#y_predict = clf.predict(x_test)
svm_score = svm_clf.score(x_test, y_test)
print('SVM Accuracy %f' %svm_score)


path = '/Users/elimiller/Desktop/AMATH482/HW4/Audio_Files/DudeLooksLikeALady.mp3'
val_data, val_rate = librosa.load(
                    path, res_type='kaiser_fast', offset=0)

validation_set, validation_labels = get_features(
                val_data, 
                val_rate, 
                'max',
                'rock')

validation_centered = validation_set - validation_set.mean(
        axis=1, keepdims=True)
validation_rep = validation_centered @ V.T[:, 0:rank]
val_score = svm_clf.score(validation_rep, validation_labels)
print('Validation Score: %f' %val_score)
