import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from homomorphic_envelop import getHeartRate
from hierarchical_clustering import get_clustering


def s1_distribution(x, size = 10000):
    mean = 122 # ms
    std_dev  = 22 # ms
    probability = norm.pdf(x, mean, std_dev)
    return probability

def s2_distribution(x):
    mean = 92 # ms
    std_dev  = 22 # ms
    probability = norm.pdf(x, mean, std_dev)
    return probability

def systolic_distribution(x, systolic_data):
    systolic_mean = np.mean(systolic_data) - 122 # mean of s1
    std_dev = np.std(systolic_data)
    probability = norm.pdf(x, systolic_mean, std_dev)
    return probability

def diastolic_distribution(x, centroid_dict, points_array, systolic_data):
    total_points = sum(len(sublist) for sublist in points_array.values())
    ratios = {key: len(sublist) / total_points for key, sublist in points_array.items()}
    
    # print("Centroid", centroid_dict)
    # print("Point Array", points_array)
    # print("Ratio", ratios)
    # print("Systole", systolic_data)
    
    
    systolic_mean = np.mean(systolic_data)
    centroid_dict = {key: value - 92 - systolic_mean for key, value in centroid_dict.items()}
    epsilon_values = {}
    
    for key, sublist in points_array.items():
        ni = len(sublist)  # Number of elements in the list
        if ni > 1:
            sigma_ci = np.std(sublist)
            # sigma_ci = (centroid_dict[key] * 0.07) + 6
            epsilon_values[key] = sigma_ci
        elif ni == 1:
            epsilon_values[key] = 25 
    
    probability = sum(
        ratios[key] * np.exp(-(x - centroid_dict[key])**2 / (2 * epsilon_values[key]**2)) / (epsilon_values[key] * np.sqrt(2 * np.pi))
        for key in centroid_dict
    )
    
    return probability

def get_distribution(centroid_array, points_array,systolic_duration):
    x_values = np.linspace(0, 1500, 3000)  # Adjust the range as needed

    # Calculate corresponding probabilities
    pdf_values = diastolic_distribution(x_values,centroid_array, points_array,systolic_duration)
    # pdf_values = systolic_distribution(x_values, systolic_duration)
    # pdf_values = systolic_distribution(x_values, systolic_duration)

    # Plot the PDF
    plt.plot(x_values, pdf_values, label='PDF')
    plt.title('Probability Density Function (PDF) of Diastole Distribution')
    # plt.title('Probability Density Function (PDF) of Systole Distribution')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig('dis_diastole.png')
    # plt.savefig('dis_systole.png')
    plt.show()
    
if __name__ == '__main__':
    # audio_file = "/home3/luharj/AiSteth/Training_Data/Yaseen21Khan Dataset/Dataset/N_New/N_New_3주기/New_N_001.wav"
    audio_file = "/home3/luharj/AiSteth/Training_Data/Circor/Normal Signal/43852_AV.wav"
    # audio_file = '/home3/luharj/AiSteth/Training_Data/New folder/14998_MV.wav'
    heart_rate, systolic_duration = getHeartRate(audio_file= audio_file)
    centroid_array, points_array = get_clustering(cluster_data= heart_rate)
    print(centroid_array)
    print(points_array)
    print(systolic_duration)
    get_distribution(centroid_array, points_array,systolic_duration)
    