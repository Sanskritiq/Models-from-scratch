import numpy as np
from queue import PriorityQueue
from tqdm import trange

from constants import *

class Cluster:
    def __init__(self, id:int = 0) -> None:
        self.id = id
        self.points = []
        
class Hierarchial:
    
    def __init__(self, distance_matrix, linkage:str, n_clusters:int = n_clusters) -> None:
        self.cluster_list = []
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.get_distance = self.single_linkage if self.linkage == 'single' else self.complete_linkage if self.linkage == 'complete' else self.average_linkage
        print('Initializing...')
        self.distance_matrix = distance_matrix
        for i in range(len(data_df)):
            cluster = Cluster(i)
            cluster.points.append(i)
            self.cluster_list.append(cluster)
        
        
    def merge_clusters(self, cluster1: Cluster, cluster2: Cluster) -> Cluster:
        new_cluster = Cluster()
        new_cluster.points = cluster1.points + cluster2.points
        new_cluster.id = min(cluster1.id, cluster2.id)
        return new_cluster
    
    def single_linkage(self, cluster1: Cluster, cluster2: Cluster) -> float:
        min_dist = float('inf')
        for point1 in cluster1.points:
            for point2 in cluster2.points:
                dist = self.distance_matrix[point1][point2]
                if dist < min_dist:
                    min_dist = dist
                    
        return min_dist
    
    def complete_linkage(self, cluster1: Cluster, cluster2: Cluster) -> float:
        max_dist = 0
        for point1 in cluster1.points:
            for point2 in cluster2.points:
                dist = self.distance_matrix[point1][point2]
                if dist > max_dist:
                    max_dist = dist
                    
        return max_dist
    
    def average_linkage(self, cluster1: Cluster, cluster2: Cluster) -> float:
        avg_dist = 0
        for point1 in cluster1.points:
            for point2 in cluster2.points:
                dist = self.distance_matrix[point1][point2]
                avg_dist += dist
                
        avg_dist /= len(cluster1.points) * len(cluster2.points)
        return avg_dist
    
    def clustering(self):
        print('Clustering...')
        
        for _ in trange(len(data_df) - self.n_clusters, desc='Hierarchial Clustering'):
            min_dist = float('inf')
            min_i = -1
            min_j = -1
            for i in range(len(self.cluster_list)):
                for j in range(i + 1, len(self.cluster_list)):
                    dist = self.get_distance(self.cluster_list[i], self.cluster_list[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_i = i
                        min_j = j
                        
            self.cluster_list.append(self.merge_clusters(self.cluster_list[min_i], self.cluster_list[min_j]))
            self.cluster_list.pop(max(min_i, min_j))
            self.cluster_list.pop(min(min_i, min_j))
            
            
if __name__ == "__main__":
    print('hey')
    hierarchial = Hierarchial('single')
    hierarchial.clustering()
    print(len(hierarchial.cluster_list))
        

