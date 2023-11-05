import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import trange
path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/cancer.csv'


data_df = pd.read_csv(path + DATA_PATH)
data = data_df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
data['cluster'] = 0
# data = data.to_numpy()
keys = data_df.keys().drop(['id', 'Unnamed: 32', 'diagnosis']).to_numpy()
keys_dict = {key: i for i, key in enumerate(keys)}
n_clusters = 2

def distance(point1, point2):
    return np.sqrt(((point1 - point2)**2).sum())

def dist_matrix(data_df):
    dist = np.zeros((len(data_df), len(data_df)))
    for i in trange(len(data_df), desc='Calculating distance matrix'):
        for j in range(len(data_df)):
            dist[i][j] = distance(data_df.iloc[i], data_df.iloc[j])
            
    dist = dist / (np.max(dist) - np.min(dist))
    return dist

if __name__=='__main__':
    print(keys_dict)
    print(keys)
