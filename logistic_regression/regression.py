import numpy as np
import pandas as pd

class Linear_regression:
    def __init__(self, X: np.array, y: np.array, epochs: int, lr: float = 0.0001) -> None:
        self.X = X
        self.y = y
        self.epochs = epochs
        self.lr = lr
        
        self.n_features = len(self.X[0])
        self.n_data = len(self.X)
        
        self.w = np.random.rand(self.n_features, self.n_data)
        self.b = np.random.rand(self.n_data, self.n_data)
        
    def prediction(self):
        # n_data, n_feature, n_feature, n_data 
        return np.matmul(self.X, self.w) + self.b
    
    def gradient(self):
        pass

class LogisticRegression:
    def __init__(self) -> None:
        pass

    