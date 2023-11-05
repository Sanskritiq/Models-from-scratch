import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import Network
from preprocessing import Dataset
from functions import Evaluation_metrics

class ModelEvaluator:
    def __init__(self, model: Network, dataset: Dataset, n_classes):
        self.model = model
        self.x_test, self.y_test = dataset.data, dataset.targets
        y_pred_classes = model.predict(self.x_test)
        self.y_pred = np.argmax(y_pred_classes, axis=1)
        self.n_classes = n_classes
        self.evaluator = Evaluation_metrics(self.y_pred, self.y_test, self.n_classes)
        self.conf_mat = self.evaluator.confusion_matrix()
        self.acc = self.evaluator.accuracy()
        print(f"Accuracy: {self.acc}")
    
    def plot_confusion_matrix(self):
        plt.imshow(self.conf_mat, cmap='Blues')
        plt.colorbar()        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        
if __name__=="__main__":
    conf_mat = np.random.randint(0, 100, size=(10,10))
    plt.imshow(conf_mat, cmap='Blues')
    plt.colorbar()        
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
       
        
    