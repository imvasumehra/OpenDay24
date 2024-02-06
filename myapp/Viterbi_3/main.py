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
    signal = signal[:100]
    states = states[:100]
    one_state = []
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
    
def list_data_path(dir_path):
    filenames = [f'{dir_path}/{name[:-4]}' for name in os.listdir(dir_path) 
                 if name.endswith('.wav')]
    return filenames


if __name__ == '__main__':
    # audio_file = "/home3/luharj/AiSteth/Training_Data/Yaseen21Khan Dataset/Dataset/N_New/N_New_3주기/New_N_001.wav"
    # filename = "/home3/luharj/AiSteth/Training_Data/New folder//49930_TV"
    
    dir_path = "/home3/luharj/AiSteth/Training_Data/New folder/"
    filenames = list_data_path(dir_path= dir_path)
    # print(len(filenames))
    filenames = [filename for filename in filenames if 'Phc' not in filename]
    # filenames = filenames[199:]
    # for index in range(0, len(filenames), 50):
        # filenamess = filenames[index: min(index + 50,len(filenames))]
    actual_labels = []
    predicted_labels = []
    for num, filename in enumerate(tqdm(filenames, desc='Files')):
        # print(filename)
        audio_file = filename + '.wav'
        mat_file = filename + '.mat'
        
        posteriors = get_features(audio_file= audio_file)
        prob_signal = posteriors[:,0] + posteriors[:,2] # This signal has sampling rate of 50 Hz
        
        heart_rate, systolic_duration = getHeartRate(original_signal= prob_signal)
        centroid_array, points_array = get_clustering(cluster_data= heart_rate)
        
        get_distribution(centroid_array= centroid_array, points_array= points_array, systolic_duration= systolic_duration)
        try :
            states_np = hsmm_viterbi(posteriors= posteriors, max_duration= 40, centroid_dict= centroid_array, points_array= points_array, systolic_data= systolic_duration)
        except :
            continue
        

        if len(np.unique(states_np)) < 4:
            print('Uniqueue len gth is less than 4', np.unique(states_np))
        label_info = loadmat(f'{filename}.mat')
        actual = (label_info['PCG_states'][0]) - 1
        
        current_value = actual[0]
        start_index = 0
        
        actual_label = []
        predicted_label = []
        
        for i in range(1, len(states_np)):
            if actual[i] != current_value:
                end_index = i
                
                # find max occurance value from index start_index to end_index from array predicted
                max_occurrence_value = np.argmax(np.bincount(states_np[start_index:end_index]))
                actual_label.append(current_value)
                predicted_label.append(max_occurrence_value)
                
                
                current_value = actual[i]
                start_index = i

        end_index = len(actual)
        max_occurrence_value = np.argmax(np.bincount(states_np[start_index:end_index]))
        actual_label.append(int(current_value))
        predicted_label.append(max_occurrence_value)
        
        actual_labels.extend(actual_label)
        predicted_labels.extend(predicted_label)
            
    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)
    
    print(actual_labels.shape)
    print(predicted_labels.shape)
    
    unique_actual_labels, counts_actual_labels = np.unique(actual_labels, return_counts=True)
    print("Actual Labels - Unique Values:")
    print(unique_actual_labels)
    print("Counts:")
    print(counts_actual_labels)

    # Print unique values and their counts for predicted labels
    unique_predicted_labels, counts_predicted_labels = np.unique(predicted_labels, return_counts=True)
    print("\nPredicted Labels - Unique Values:")
    print(unique_predicted_labels)
    print("Counts:")
    print(counts_predicted_labels)
    
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(actual_labels), yticklabels=np.unique(actual_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.savefig(f'cm.png')
    plt.show()
    plt.clf()
                
        
    # print(posteriors.shape)
    
    # get Probablity Signal *************
    # plt.figure(figsize=(15, 4))
    # plt.plot(prob_signal, label='S1', color='blue')
    # plt.title('RNN Output Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.savefig("rnn_op.png")
    # plt.show()
    
    
    # print(prob_signal.shape) 
    
    
    
    
    
    
    
    
    # print(centroid_array, points_array)
    
    
    #
    # save_signal_image(audio_file= audio_file, states= states_np)
    