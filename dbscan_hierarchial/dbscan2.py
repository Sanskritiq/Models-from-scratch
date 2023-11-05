from constants import *
from tqdm import tqdm
import copy


class Cluster:
    def __init__(self, id: int = 0) -> None:
        self.id = id
        self.points = []


class DBSCAN:
    def __init__(self, distance_matrix, eps, MinPoints: int = 3) -> None:
        self.distance_matrix = distance_matrix
        self.eps = eps
        self.MinPoints = MinPoints
        self.visited = []
        self.cluster_list = []
        self.outliers = []
        self.cores = []
        self.borders = []
        self.neighors = {}

    def region_scan(self, indx):
        neighbors = []
        for i in range(len(data_df)):
            if self.distance_matrix[indx][i] <= self.eps and self.distance_matrix[indx][i] != 0:
                neighbors.append(i)
        return neighbors

    def merge_clusters(self, indx: int, neighbors_cluster_indx: list):
        new_cluster = Cluster()
        for cluster_indx in neighbors_cluster_indx:
            new_cluster.id = min(new_cluster.id, self.cluster_list[cluster_indx].id)
            new_cluster.points += self.cluster_list[cluster_indx].points
            new_cluster.points.append(indx)
        
        for cluster_indx in sorted(neighbors_cluster_indx, reverse=True):
            del self.cluster_list[cluster_indx]

        new_cluster.id = len(self.cluster_list)
        self.cluster_list.append(new_cluster)

    def assign_cluster(self, indx: int, neighbors: list):
        neighbors_cluster_indx = []
        for i in neighbors:
            if i not in self.outliers:
                for j in range(len(self.cluster_list)):
                    cluster = self.cluster_list[j]
                    if i in cluster.points:
                        self.visited.append(i)
                        if j not in neighbors_cluster_indx:
                            neighbors_cluster_indx.append(j)

        if len(neighbors_cluster_indx) == 0:
            cluster = Cluster()
            cluster.id = len(self.cluster_list)
            cluster.points.append(indx)
            self.cluster_list.append(cluster)
            return

        self.merge_clusters(indx, neighbors_cluster_indx)

    def clustering(self):
        pbar = tqdm(total=len(data_df))
        for indx in range(len(data_df)):
            neighbors = self.region_scan(indx)
            self.neighors[indx] = neighbors
            if len(neighbors) >= self.MinPoints:
                self.cores.append(indx)
                self.visited.append(indx)
                pbar.update(1)
                pbar.refresh()
               
        for indx in self.cores:
            neighbors = self.neighors[indx]
            for neighbor in neighbors:
                if neighbor not in self.visited:
                    self.borders.append(neighbor)
                    self.visited.append(neighbor)
                    pbar.update(1)
                    pbar.refresh()
                    
        for indx in range(len(data_df)):
            if indx not in self.visited:
                self.outliers.append(indx)
                self.visited.append(indx)
                pbar.update(1)
                pbar.refresh()
                
        pbar.n = len(data_df) - len(self.outliers)
        pbar.refresh()
        self.visited = copy.copy(self.outliers)
        for indx in self.cores:
            if indx not in self.visited:
                self.visited.append(indx)
                has_cluster = False
                for cluster in self.cluster_list:
                    if indx in cluster.points:
                        has_cluster = True
                        break
                if not has_cluster:
                    self.assign_cluster(indx, self.neighors[indx])
                    pbar.update(len(self.neighors[indx]))
                    pbar.refresh()
                    
        
        pbar.n = len(data_df)
        pbar.refresh()

            
            
        
            


if __name__ == '__main__':
    distance_matrix = dist_matrix(data)
    clustering = DBSCAN(distance_matrix, eps=0.01, MinPoints=6)
    print(distance_matrix)
    clustering.clustering()
    print(len(clustering.cluster_list))
    print(len(clustering.cluster_list[0].points))
    # print(clustering.cluster_list[1].points)
    print(len(clustering.outliers))
    print(len(clustering.cores))
    print(len(clustering.borders))
