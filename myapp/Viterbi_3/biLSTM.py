from features import get_features
from tqdm import tqdm
import os
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def list_data_path(dir_path):
    filenames = [f'{dir_path}/{name[:-4]}' for name in os.listdir(dir_path) 
                 if name.endswith('.wav')]
    return filenames

if __name__ == '__main__':
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
        label_info = loadmat(f'{filename}.mat')
        actual = (label_info['PCG_states'][0]) - 1
        
        predicted = np.argmax(posteriors, axis=1)
        
        current_value = actual[0]
        start_index = 0
        
        actual_label = []
        predicted_label = []
        
        for i in range(1, len(predicted)):
            if actual[i] != current_value:
                end_index = i
                
                # find max occurance value from index start_index to end_index from array predicted
                max_occurrence_value = np.argmax(np.bincount(predicted[start_index:end_index]))
                actual_label.append(current_value)
                predicted_label.append(max_occurrence_value)
                
                
                current_value = actual[i]
                start_index = i

        end_index = len(actual)
        max_occurrence_value = np.argmax(np.bincount(predicted[start_index:end_index]))
        actual_label.append(int(current_value))
        predicted_label.append(max_occurrence_value)
        
        actual_labels.extend(actual_label)
        predicted_labels.extend(predicted_label)
        
    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)
    
    print(actual_labels.shape)
    print(predicted_labels.shape)
    
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(actual_labels), yticklabels=np.unique(actual_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.savefig(f'rnn.png')
    plt.show()
    plt.clf()