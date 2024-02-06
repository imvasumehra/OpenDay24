from homomorphic_envelop import getHeartRate
from hierarchical_clustering import get_clustering
import librosa
import numpy as np
from features import get_features
from viterbi import hsmm_viterbi
import librosa.display
import matplotlib.pyplot as plt
from distributions import get_distribution
from tqdm import tqdm
from scipy.io import loadmat
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

def save_signal_image(audio_file, states):
    signal, sr = librosa.load(audio_file, sr=50)
    signal = signal[5:100]
    states = states[5:100]
    time = np.arange(0, len(signal)) / sr
    s1 = np.unique(states)
    print(s1)
    time = np.arange(0, len(signal)) / sr
    observation_marker_colors = ['red' if s2 == s1[0] else 'green' if s2 == s1[1] else 'blue' if s2 == s1[2] else 'yellow' for s2 in states]
    plt.figure(figsize=(14, 8))
    for i in range(len(signal) - 1):
        plt.plot([time[i], time[i + 1]], [signal[i], signal[i + 1]], color=observation_marker_colors[i])


    plt.title('Audio Signal with States')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.savefig("label.png")
    plt.show()
    plt.clf()
    
    plt.figure(figsize=(14, 8))
    plt.plot(time, signal)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.savefig("signal.png")
    plt.show()

posteriors = get_features(audio_file='./../uploads/test_file.wav')
prob_signal = posteriors[:,0] + posteriors[:,2] # This signal has sampling rate of 50 Hz
        
heart_rate, systolic_duration = getHeartRate(original_signal= prob_signal)
centroid_array, points_array = get_clustering(cluster_data= heart_rate)
        
get_distribution(centroid_array= centroid_array, points_array= points_array, systolic_duration= systolic_duration)
try :
    states_np = hsmm_viterbi(posteriors= posteriors, max_duration= 40, centroid_dict= centroid_array, points_array= points_array, systolic_data= systolic_duration)
except :
    pass

save_signal_image('./../uploads/test_file.wav', states_np)