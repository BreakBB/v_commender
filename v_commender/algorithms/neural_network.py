import numpy as np


class NeuralNetwork:
    training_data = []
    training_votes = []
    weights_1 = 0
    weights_2 = 0
    layer_1 = None
    layer_2 = None
    output = []

    def __init__(self, x, y):
        self.training_data = x
        neurones_per_layer = 4
        self.weights_1 = np.random.rand(self.training_data.shape[1], neurones_per_layer)
        self.weights_2 = np.random.rand(neurones_per_layer, 1)
        self.training_votes = y
        self.output = np.zeros(y.shape)

    def feed_forward(self):
        self.layer_1 = self.sigmoid(np.dot(self.training_data, self.weights_1))
        self.layer_2 = self.sigmoid(np.dot(self.layer_1, self.weights_2))
        return self.layer_2

    def back_prop(self):
        d_weights2 = np.dot(self.layer_1.T,
                            2 * (self.training_votes - self.output) * self.sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.training_data.T,
                            np.dot(2 * (self.training_votes - self.output) * self.sigmoid_derivative(self.output),
                                   self.weights_2.T) * self.sigmoid_derivative(self.layer_1))

        self.weights_1 += d_weights1
        self.weights_2 += d_weights2

    def predict(self, predict_item):
        layer_1 = self.sigmoid(np.dot(predict_item, self.weights_1))
        return self.sigmoid(np.dot(layer_1, self.weights_2))

    def train(self):
        self.output = self.feed_forward()
        self.back_prop()

    # Activation function
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    # Derivative of sigmoid
    def sigmoid_derivative(self, p):
        return p * (1 - p)

    # Mean sum squared loss
    def get_loss(self):
        return np.mean(np.square(self.training_votes - self.feed_forward()))
