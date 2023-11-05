import numpy as np
import pandas as pd
DATA_PATH = 'cancer.csv'

import matplotlib.pyplot as plt

data_df = pd.read_csv(DATA_PATH)
data = data_df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
data['cluster'] = 0
data = data.to_numpy()
keys = data_df.keys().drop(['id', 'Unnamed: 32', 'diagnosis']).to_numpy()
keys_dict = {key: i for i, key in enumerate(keys)}
n_clusters = 2 
n_iters = 6
        
class Cluster:
    
    def __init__(self, label) -> None:
        self.center = np.random.choice(data.shape[0], replace=False)
        self.label = label
        
    def update_center_kmeans(self):        
        self.center = np.mean(data[np.where(data[:, len(keys)]==self.label)], axis=0)
        
    def update_center_kmedoids(self):        
        self.center = np.median(data[np.where(data[:, len(keys)]==self.label)], axis=0)
        
    def calculate_distance(self, point):        
        dist = np.linalg.norm(point - self.center)            
        return dist