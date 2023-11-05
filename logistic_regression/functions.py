import numpy as np
import pandas as pd

class Evaluation_metrics:
    def __init__(self, y_pred, y_truth, n_classes) -> None:
        self.y_pred = y_pred
        self.y_truth = y_truth
        self.n_classes = n_classes
        
    def confusion_matrix(self):
        self.conf_mat = np.zeros([self.n_classes, self.n_classes])
        for i in range(len(self.y_pred)):
            self.conf_mat[self.y_truth[i]][self.y_pred[i]] += 1
            
        return self.conf_mat
        
    def accuracy(self):
        self.acc = 0
        for i in range(self.n_classes):
            self.acc += self.conf_mat[i][i]
        self.acc /= len(self.y_pred)
        return self.acc
    
    def cross_entropy():
        pass
    

if __name__ == '__main__':
    from preprocessing import Dataset, load
    labels, pixels = load('archive/mnist_train.csv')
    train = Dataset(labels, pixels)
    labels, pixels = load('archive/mnist_test.csv')
    test = Dataset(labels, pixels)
