# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:05:19 2021

@author: 91987
"""

import librosa
import soundfile
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv




# Extract Features (mfcc, chroms, mel) from a sound file
def extract_features(file_name, mfcc, chroma, mel):
	with soundfile.SoundFile(file_name) as f:
		X = f.read(dtype = "float32")
		sampleRate = f.samplerate
		if chroma:
			stft = np.abs(librosa.stft(X))
		result = np.array([])
		if mfcc:
			mfccs = np.mean(librosa.feature.mfcc(y=X, sr = sampleRate, n_mfcc = 60).T, axis=0)
			result = np.hstack((result, mfccs))
		if chroma:
			chromas = np.mean(librosa.feature.chroma_stft(S = stft, sr = sampleRate).T, axis=0)
			result = np.hstack((result, chromas))
		if mel:
			mels = np.mean(librosa.feature.melspectrogram(X, sr = sampleRate).T, axis=0)
			result = np.hstack((result, mels))
	return result	


def predictions(f):

	# TESTING ON A AUDIO FILE
	
    modelName = "speechRecogModel2.sav"
    loaded_model = pickle.load(open(modelName, 'rb')) # Loading the model file from storage
 	# file = os.path.join("speech_emotion_recognition_ravdess_data\Actor_13","03-01-07-01-01-01-13.wav")
   
    feature = extract_features(f, mfcc = True, chroma = True, mel = True)
    print(feature)
	# y_pre = model.predict([feature])
    feature = feature.reshape(1,-1)
    prediction = loaded_model.predict(feature)
    print(prediction)
	# print(y_pre)5
    return(prediction)
    

def record():
 
    # Sampling frequency
    freq = 4000
    
    # Recording duration
    duration = 3
    
    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
    				samplerate=freq, channels=1)
    
    # Record audio for the given number of seconds
    sd.wait()
    
    f = "recording.wav"
    
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write(f, freq, recording)
    
    return f
    



# predictions(os.path.join("D:\emotions\SERkit","recording2.wav")) 
# predictions(os.path.join("D:\emotions\SERkit\speech-emotion-recognition-ravdess-data\Actor_03","03-01-03-01-01-01-03.wav"))