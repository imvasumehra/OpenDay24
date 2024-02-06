import librosa
import numpy as np 
import os

def denoise():
    file = os.listdir('./uploads/')[0]
    _, sr = librosa.load('./uploads/' + file, sr = None)
    return sr
