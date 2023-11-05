import numpy as np
import matplotlib.pyplot as plt

from preprocessing import Utily

class LinearRegression:
    def __init__(self, data_x, data_y) -> None:
        self.data_x = data_x
        self.data_y = data_y
        
        self.x_mean = np.mean(data_x)
        self.y_mean = np.mean(data_y)
        
        self.x_var = np.var(data_x)
        self.x_y_covar = np.cov(data_x, data_y)[0][1]
        
        self.theta1 = self.x_y_covar/self.x_var
        self.theta2 = self.y_mean - self.theta1*self.x_mean
        
    def predict(self, x):
        return self.theta1*x + self.theta2


    def plot(self, test_data_x, test_data_y):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
        # add scatter plot 
        axes[0].scatter(self.data_x, self.data_y, color='red', marker = 'o')
        x = np.array([-10,1300])
        y = self.theta1*x + self.theta2
        axes[0].plot(x, y, label='Line', color='blue')
        axes[0].set_title('Training set')
        axes[0].set_xlabel('X-axis')
        axes[0].set_ylabel('Y-axis')
        # axes[0].legend()
        # add 

        axes[1].scatter(test_data_x, test_data_y, color='red', marker = 'o')
        axes[1].plot(x, y, label='Line', color='blue')
        axes[1].set_title('Test set')
        axes[1].set_xlabel('X-axis')
        axes[1].set_ylabel('Y-axis')
        # axes[1].legend()

        plt.show()

if __name__ == '__main__':
    import os
    path = os.path.dirname(os.path.abspath(__file__)) + "/"
    x,y = Utily.load(path + "data.tsv")
    lr = LinearRegression(x,y)

    _, test_x = Utily.load(path + "test_data.tsv")
    test_y = []
    for i in range(len(test_x)):
        test_y.append(lr.predict(test_x[i]))
    lr.plot(test_x, test_y)

    test_y = np.array(test_y)
    sunday_percent = (test_y / test_x) * 100
    average_percentage = sunday_percent.mean()
    minimum_percentage = sunday_percent.min()
    maximum_percentage = sunday_percent.max()
    print("average_percentage : ", average_percentage)
    print("minimum_percentage : ", minimum_percentage)
    print("Maximum_percentage : ", maximum_percentage)