from tqdm import trange
import numpy as np
from math import e
from preprocessing import Dataset
from functions import Evaluation_metrics

class Layer:
    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def _forward(self, data):
        raise NotImplementedError
    
    def _backward(self, loss, lr):
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"[ {self.__name__} ]: {self.input_shape} -> {self.output_shape}]"

class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def _forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def _backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error
        
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class LogisticLayer(Layer):
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__(input_shape, output_shape)
        self.__name__ = "Logistic Layer"
        self.weight = np.random.rand(input_shape, output_shape)*0.01 # [A, B]
        self.bias = np.random.rand(1, output_shape)*0.01
    
    def _forward(self, data):
        self.input_data = data
        return 1 / ( 1 + np.exp(-np.dot(data, self.weight) + self.bias))
    
    def _backward(self, loss, lr):
        # loss = [N, B]
        inp_error = np.dot(loss, self.weight.T)
        # [784, 1] dot [N, B] -> [784, B]
        weight_error = np.dot(self.input_data.reshape(-1, 1), loss)

        # adjust weights and biases
        self.weight -= lr * weight_error
        self.bias -= lr * loss

        return inp_error


class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def _forward(self, input):
        self.input = input
        return self.activation(input)
    
    def _backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)

    def __str__(self) -> str:
        return f"[ Activation ]: {self.__name__}"

    
class SigmoidActivationLayer(ActivationLayer):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    
class TanhActivation(ActivationLayer):
    def __init__(self, input_shape) -> None:
        self.__name__ = "Tanh"
        self.activation = lambda x: np.tanh(x)
        self.activation_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(self.activation, self.activation_prime)

class Network:
    def __init__(self, error, error_prime) -> None:
        self.layers = []
        self.error = error
        self.error_prime = error_prime

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, epochs, lr):
        samples = len(x_train)
        self.lr = lr

        for epoch in range(epochs):
            error = 0
            for j in trange(samples):
                # forward
                output = x_train[j].reshape(1, -1)
                y_true = y_train[j]
                for layer in self.layers:
                    output = layer._forward(output)
                
                # error (display purpose only)
                error += self.error(y_true, output)

                # backward
                output_error = self.error_prime(y_true, output)
                for layer in reversed(self.layers):
                    output_error = layer._backward(output_error, lr)
    
            error /= len(x_train)
            print('%d/%d, error=%f' % (epoch + 1, epochs, error))

    def predict(self, data_x):
        for layer in self.layers:
            data_x = layer._forward(data_x)
        return data_x
    
    def evalute(self, x_test, y_test, n_classes):
        print("Evaluating...")
        y_pred = self.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        # print(y_pred[0], y_test[0])
        evaluator = Evaluation_metrics(y_pred, y_test, n_classes)
        conf_mat = evaluator.confusion_matrix()
        acc = evaluator.accuracy()
        print(f"Accuracy: {acc*100}%")
        # print(f"Confusion Matrix: \n{conf_mat}")
        return conf_mat, acc
    
class Error:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size

    @staticmethod
    def sse(y_true, y_pred):
        return 0.5 * np.sum(np.power(y_true - y_pred, 2))

    @staticmethod
    def sse_prime(y_true, y_pred):
        return y_pred - y_true
    

if __name__ == '__main__':
    epochs = 30
    learning_rate = 0.1

    # (x_train, y_train), (x_test, y_test) = mnist_loader()
    train = Dataset('archive/mnist_train.csv')
    test = Dataset('archive/mnist_test.csv')
    x_train, y_train = train.data, train.targets
    x_test, y_test = test.data, test.targets
    # print(train_x.shape, train_y.shape)
    # x_train = x_train[:10000]
    # y_train = y_train[:10000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    network = Network(Error.mse, Error.mse_prime)
    # network.add(FlattenLayer(input_shape=(28, 28)))
    network.add(LinearLayer(28 * 28, 10))
    network.add(SigmoidActivationLayer())
    network.fit(x_train, y_train, epochs, learning_rate)

    conf_mat, acc = network.evalute(x_test, y_test, 10)
    
    ## plot confusion matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.imshow(conf_mat, cmap='Blues')
    plt.colorbar()        
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')

    
