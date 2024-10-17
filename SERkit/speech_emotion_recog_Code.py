# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 22:35:12 2021

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


# Emotions in the RAVDESS dataset
emotions = {'01':'neutral',
			'02':'calm',
			'03':'happy',
			'04':'sad',
			'05':'angry',
			'06':'fearful',
			'07':'disgust',
			'08':'surprised'}

observed_emotions = ["angry", "happy", "neutral", "disgust", "sad"]
# leave = ["surprised"]

# Load the data and extract features for each sound file
def load_data(test_size = 0.2):
	x, y = [],[]
	for file in glob.glob("D:\emotions\SERkit\speech-emotion-recognition-ravdess-data\Actor*\*.wav"):
		file_name = os.path.basename(file)
		print(file_name, end='\n')
		emotion = emotions[file_name.split('-')[2]]
		if emotion not in observed_emotions:
 			continue
# 		if emotion in leave:
# 			continue
		feature = extract_features(file, mfcc = True, chroma = True, mel = True)
		x.append(feature)
		y.append(emotion)
	return train_test_split(np.array(x), y, test_size = test_size, random_state = 9)
			
# Split the dataset
print("loading dataset")
x_train, x_test, y_train, y_test = load_data(test_size=0.25) 

print(x_train)

print(x_train[0].shape, x_test[0].shape)

print(f'Features Extracted: {x_train.shape[1]}')

model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (300,), learning_rate = "adaptive", max_iter = 500)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_pred)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
print(y_pred)

f1 = f1_score(y_test, y_pred, average = None)
print(f1)

df = pd.DataFrame({"Actual":y_test, "Prediciton":y_pred})
print(df.head(20))

# Storing Model as pickle
# Writing different model files to file
with open("speechRecogModel1.sav", "wb") as f:
	pickle.dump(model, f)


# TESTING ON A AUDIO FILE
modelName = "speechRecogModel.sav"
loaded_model = pickle.load(open(modelName, 'rb')) # Loading the model file from storage
file = os.path.join("D:\emotions\SERkit\speech-emotion-recognition-ravdess-data\Actor_13","03-01-07-01-01-01-13.wav")
feature = extract_features(file, mfcc = True, chroma = True, mel = True)
y_pre = model.predict([feature])
feature = feature.reshape(1,-1)
prediction = loaded_model.predict(feature)
print(prediction)
# print(y_pre)

# print(feature)


