import numpy as np
from tqdm import tqdm
import json
import os

def initialize_cluter(data):
    centroid_dict = {i: point for i, point in enumerate(data)}
    points_array = {i: [point] for i, point in enumerate(data)}
    
    n = len(data)
    dissimilarity_array = np.zeros((n, n))
    
    # for i in tqdm(range(n), desc="Calculating dissimilarity"):
    for i in range(n):
        for j in range(i+1, n):
            dissimilarity = np.abs(data[i] - data[j])
            dissimilarity_array[i, j] = dissimilarity
            dissimilarity_array[j, i] = dissimilarity
            
    return centroid_dict, points_array, dissimilarity_array

def getCentroid(cluster1, cluster2, centroid_dict, points_array):
    n1 = len(points_array[cluster1])
    n2 = len(points_array[cluster2])
    
    c1 = centroid_dict[cluster1]
    c2 = centroid_dict[cluster2]
    
    return ((n1 * c1) + (n2 * c2)) / (n1 + n2)

def getDissimilarity(cluster1, cluster2, cluster3, points_array, dissimilarity_array):
    # print(points_array)
    n1 = len(points_array[cluster1])
    n2 = len(points_array[cluster2])
    n3 = len(points_array[cluster3])
    
    dissimilarity1 = (n1 + n3) * dissimilarity_array[cluster1, cluster3]
    dissimilarity2 = (n2 + n3) * dissimilarity_array[cluster2, cluster3]
    dissimilarity3 = (n1 + n2) * dissimilarity_array[cluster1, cluster2]
    
    return (dissimilarity1 + dissimilarity2 - dissimilarity3) / (n1 + n2 + n3)

def merge_cluster(centroid_dict, points_array, dissimilarity_array):
    # Get the Minimum dissimilarity values from n*n array
    min_index = np.unravel_index(np.argmin(np.ma.masked_where(np.eye(len(dissimilarity_array)), dissimilarity_array)), dissimilarity_array.shape)
    
    #Get two cluster for merge
    cluster1, cluster2 = min_index
    # print(cluster1, cluster2)
    dis_value = dissimilarity_array[cluster1, cluster2]
    
    #Merge data points of two cluster
    merge_cluster_point = points_array[cluster1] + points_array[cluster2]
    #Get centroid of new cluster
    merge_cluster_centroid = getCentroid(cluster1= cluster1, cluster2= cluster2, centroid_dict= centroid_dict, points_array= points_array)
    
    old_dissimilarity_array = np.array(dissimilarity_array)
    old_points_array = points_array.copy()
    old_centroid_dict = centroid_dict.copy()
    
    #250 is threasold
    if dis_value > 262:
        return dis_value, old_centroid_dict, old_points_array, old_dissimilarity_array
    
    #Update centroid and Points dict
    centroid_dict[len(centroid_dict)] = merge_cluster_centroid
    points_array[len(points_array)] = merge_cluster_point
    
   
    #Delete Individual clsuter entry
    del centroid_dict[cluster1]
    del centroid_dict[cluster2]
    del points_array[cluster1]
    del points_array[cluster2] 
    dissimilarity_array = np.delete(dissimilarity_array, [cluster1, cluster2], axis=0)
    dissimilarity_array = np.delete(dissimilarity_array, [cluster1, cluster2], axis=1)
    
    #Assigining the continuous keys
    new_keys = {i: centroid_dict[key] for i, key in enumerate(sorted(centroid_dict.keys()))}
    centroid_dict.clear()
    centroid_dict.update(new_keys)

    new_keys = {i: points_array[key] for i, key in enumerate(sorted(points_array.keys()))}
    points_array.clear()
    points_array.update(new_keys)
    
    #Update the disSimilarity array
    new_dissimilarity_array = []
    for cluster in range(len(old_dissimilarity_array)):
        if cluster == cluster1 or cluster == cluster2:
            continue
        new_dissimilarity_array.append(getDissimilarity(cluster1= cluster1, 
                                                        cluster2= cluster2, 
                                                        cluster3= cluster, 
                                                        points_array= old_points_array, 
                                                        dissimilarity_array= old_dissimilarity_array))

    
    dissimilarity_array = np.vstack([dissimilarity_array, new_dissimilarity_array])
    new_dissimilarity_array.append(0)
    
    dissimilarity_array = np.hstack([dissimilarity_array, np.array(new_dissimilarity_array)[:, np.newaxis]])
    return dis_value, centroid_dict, points_array, dissimilarity_array

def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_dict_as_json(centroid_dict, ratios, epsilon_values):
    # Convert dictionaries to JSON serializable format
    folder_path = '/home3/luharj/AiSteth/Code/Segmentation/dict'
    centroid_dict_serializable = {k: convert_to_json_serializable(v) for k, v in centroid_dict.items()}
    ratios_serializable = convert_to_json_serializable(ratios)
    epsilon_values_serializable = convert_to_json_serializable(epsilon_values)

    # Save centroid_dict to a file in the specified folder
    centroid_dict_file_path = os.path.join(folder_path, 'centroid_dict.json')
    with open(centroid_dict_file_path, 'w') as file:
        json.dump(centroid_dict_serializable, file, indent=2)

    # Save ratios to a file in the specified folder
    ratios_file_path = os.path.join(folder_path, 'ratios.json')
    with open(ratios_file_path, 'w') as file:
        json.dump(ratios_serializable, file, indent=2)

    # Save epsilon_values to a file in the specified folder
    epsilon_values_file_path = os.path.join(folder_path, 'epsilon_values.json')
    with open(epsilon_values_file_path, 'w') as file:
        json.dump(epsilon_values_serializable, file, indent=2)

    # Print a message indicating that the dictionaries are saved
    print(f"Dictionaries are saved successfully")
    
def get_clustering(cluster_data):
    centroid_dict, points_array, dissimilarity_array = initialize_cluter(cluster_data)
    min_value = 0
    while(min_value <= 262) and len(centroid_dict) > 1:
        min_value, centroid_dict, points_array, dissimilarity_array = merge_cluster(centroid_dict= centroid_dict, points_array= points_array, dissimilarity_array= dissimilarity_array)
        # print(min_value, len(centroid_dict))
    total_points = sum(len(sublist) for sublist in points_array.values())
    ratios = {key: len(sublist) / total_points for key, sublist in points_array.items()}
    epsilon_values = {}

    # print(points_array, ratios)
    for key, sublist in points_array.items():
        ni = len(sublist)  # Number of elements in the list
        if ni > 1:
            sigma_ci = np.std(sublist)
            epsilon_values[key] = sigma_ci
        elif ni == 1:
            epsilon_values[key] = 25  # 25 ms if ni = 1
            
    # print("Success")
    # save_dict_as_json(centroid_dict = centroid_dict, ratios= ratios, epsilon_values= epsilon_values)
    return centroid_dict, points_array

if __name__ == "__main__":
    arr = [650, 670, 680, 645, 820, 830, 890, 900, 1200, 1150, 870, 700, 645]
    get_clustering(arr)