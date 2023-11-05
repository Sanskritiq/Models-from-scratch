import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *

class Clustering:
    
    def __init__(self, mode = 'kmedoids') -> None:
        self.clusters = []
        for label in range(n_clusters):
            self.clusters.append(Cluster(label))
        self.mode = mode
        
    def assign_clusters(self):        
        for i in range(len(data)):
            distances = []
            point = data[i]
            for cluster in self.clusters:
                dist = cluster.calculate_distance(point)
                distances.append(dist)
            label = np.argmin(distances)
            data[i][len(keys)] = label
            
    def update_centers(self):
        for cluster in self.clusters:
            if self.mode == 'kmeans':
                cluster.update_center_kmeans()
            elif self.mode == 'kmedoids':
                cluster.update_center_kmedoids()
                
    def run(self):
        for _ in range(n_iters):
            self.assign_clusters()
            self.update_centers()