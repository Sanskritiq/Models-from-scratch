# from keras.utils import to_categorical
import pandas as pd
import numpy as np

def to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class Dataset:
    def __init__(self, filename) -> None:
        labels, pixels = self.load(filename)
        self.targets = np.array(labels)
        self.data = np.array(self.normalize(pixels))
        self.targets = to_categorical(self.targets)
        
    def get_item(self, indx):
        return self.data[indx], self.targets[indx]
                
    def load(self, path):
        df = pd.read_csv(path)
        labels = df.iloc[:, 0]
        pixels = df.iloc[:, 1:]
        return labels, pixels
    
    def normalize(self, x):
        x /= 255
        return x

def mnist_loader() :
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    y_train = y_train

    x_test = x_test.astype('float32')
    x_test /= 255
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

if __name__=='__main__':
    
    test = Dataset('archive/mnist_test.csv')
    print(test.data[0][10])
    print(test.targets[0])
