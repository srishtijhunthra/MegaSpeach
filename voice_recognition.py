#!/usr/bin/env python
# coding: utf-8

# In[6]:


import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write


# In[7]:


model = pickle.load(open('model_weight/model_1.h5', 'rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


# In[9]:


def recorder(name):
    fs = 16000  # Sample rate
    seconds = 3  # Duration of recording                                            (CAN BE CHANGED)

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('./static/{}.wav'.format(name), fs, myrecording)  # Save as WAV file 


# ## READING INPUT

# In[ ]:





# In[10]:


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        #print(sample_rate)
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# In[11]:


def load_data(name):
    x=[]
    for file in glob.glob("./static/{}.wav".format(name)):
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return np.array(x)


# In[12]:


def predictor(name):
    input_x = load_data(name)
    result = model.predict(input_x)[0]
    #print('\n\n\n\n'+result)
    return result


# In[13]:


def display(name):
    result = predictor(name)
    dataframe = pd.read_csv("./storage/data_moods.csv")
    dataframe = dataframe.drop(['id','length','danceability','acousticness','energy','instrumentalness','liveness','valence','loudness','speechiness','tempo','key','time_signature'], axis = 1)
    music_list = dataframe[dataframe['mood']== result].reset_index().drop(['index'],axis=1)
    return music_list


# In[14]:



# In[ ]:




