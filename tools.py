
# normalize the data to [0,1]
# input parameter: data, list like [[x,y],[x,y],...]
# function:
# 1. normalize the input data to [0,1];
# 2. save the mapping relationship between the original data and the normalized data.
# return: normalized data, mapping relationship

import math
import numpy as np
import yaml

class HyperparametersConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        # Iterate through keys in the loaded YAML file and set them as attributes
        for key, value in config_data.items():
            setattr(self, key, value)
    
    def has_key(self, key):
        return hasattr(self, key)


# load config.ini file
# return a dict
def load_config(config_file_path):
    config_dict={}
    with open(config_file_path,'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith('#'):
                continue
            if len(line)==0:
                continue
            if '=' not in line:
                continue
            key,value=line.split('=')
            config_dict[key.strip()]=value.strip()
    return config_dict



# param
#   sorted_tsp_data: 2d np array, the sorted tsp data, each row is a point, each column is a coordinate
# pay attention: the distance should include the distance from the last point to the first point
def TSP_tour_distance(sorted_tsp_data):
    total_distance=0
    for i in range(len(sorted_tsp_data)-1):
        total_distance+=int(np.linalg.norm(sorted_tsp_data[i+1]-sorted_tsp_data[i])+0.5)
    total_distance+=int(np.linalg.norm(sorted_tsp_data[0]-sorted_tsp_data[-1])+0.5)
    return total_distance


# param data: list like [[x,y],[x,y],...]
def normalize(data):
    # get the max and min value of each column
    max_value = np.max(data,axis=0)
    min_value = np.min(data,axis=0)
    # get the range of each column
    range_value = max_value - min_value
    # get the normalized data
    normalized_data = (data - min_value) / range_value
    
    # construct the mapping between the original data and the normalized data, so that we can recover the original data from the normalized data
    mapping = {}
    for i in range(len(data)):
        mapping[tuple(normalized_data[i])] = tuple(data[i])


    return normalized_data, mapping


def cheapest_insertion(tour,depot):
    r=tour.copy()
    # calculate the total distance of the original tour
    total_distance = TSP_tour_distance(r)
    # print('total_distance:',total_distance)

    # calculate the distance between depot and each point in r
    distance = np.linalg.norm(r-depot,axis=1)
    # print('distance:',distance)

    best_index = 0
    best_distance = 100000000
    for i in range(len(r)):
        # calculate the total distance of the new tour after inserting depot into r at index i
        if i==0:
            new_distance = total_distance + math.ceil(np.linalg.norm(r[-1]-depot)) + math.ceil(np.linalg.norm(r[i]-depot)) - math.ceil(np.linalg.norm(r[-1]-r[i]))
        else:
            new_distance = total_distance + math.ceial(np.linalg.norm(r[i-1]-depot)) + math.ceil(np.linalg.norm(r[i]-depot)) - math.ceil(np.linalg.norm(r[i-1]-r[i])) 
        # print('new_distance:',new_distance)
        if new_distance < best_distance:
            best_distance = new_distance
            best_index = i

    # insert depot into r at the best index
    r = np.insert(r,best_index,depot,axis=0)
    # print('r:',r)
    return r,total_distance,new_distance

def test():
    data = []
    for i in range(10):
        x = np.random.randint(0,10)
        y = np.random.randint(0,10)
        data.append([x,y])
    
    normalized_data,mapping=normalize(data)
    print('data:',data)
    print('normalized_data:',normalized_data)
    print('mapping:',mapping)

    print("------loading original data by normalized data------")
    # load original data from normalized data
    t=tuple(normalized_data[3])
    print('t:',t)
    print('mapping[t]:',mapping[t])




# main
if __name__ == '__main__':
    test()