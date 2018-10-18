import numpy as np


class Model:
    def __init__(self, hidden_layer_size=5, learning_rate=0.1):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.w_ih = None
        self.w_ho = None
        self.z_i = None
        self.z_h = None
        self.z_o = None
        self.df_h = None
        self.d_h = None
        self.d_o = None
        self.error = None
        self.average_error = np.inf

        self.create_weights()

    def create_weights(self):
        bound = 5
        self.w_ih = np.random.uniform(-bound, bound, (5, self.hidden_layer_size))
        self.w_ho = np.random.uniform(-bound, bound, (self.hidden_layer_size + 1, 3))

    def forward_propagate(self, data):
        self.z_i = np.insert(data, 4, 1, axis=1)
        s_h = self.z_i.dot(self.w_ih)
        self.df_h = (sigmoid(s_h, deriv=True)).T
        self.z_h = np.insert(sigmoid(s_h), self.hidden_layer_size, 1, axis=1)
        s_o = self.z_h.dot(self.w_ho)
        self.z_o = softmax(s_o)

    def backward_propagate(self, answers):
        self.d_o = (cross_entropy(self.z_o, answers, deriv=True)).T
        self.d_h = self.df_h * (self.w_ho[:-1, :].dot(self.d_o))

    def update_weights(self):
        self.w_ih += -1 * self.learning_rate * (self.d_h.dot(self.z_i)).T
        self.w_ho += -1 * self.learning_rate * (self.d_o.dot(self.z_h)).T

    def calculate_error(self, answers):
        self.error = cross_entropy(self.z_o, answers)
        self.average_error = np.average(self.error)

    def clear(self):
        self.z_i = None
        self.z_h = None
        self.z_o = None
        self.df_h = None
        self.d_h = None
        self.d_o = None
        self.error = None
        self.average_error = np.inf


# Logistic sigmoid function
def sigmoid(x, deriv=False):
    y = 1.0 / (1 + np.exp(-x))
    if deriv is False:
        return y
    else:
        return y * (1 - y)


# Linear function
def linear(x, deriv=False):
    if deriv is False:
        return x
    else:
        return 1


# Softmax function (Only works for 2d arrays for now)
# Returns 1, which is multiplied by the gradient from cross entropy wrt z_o
def softmax(x, deriv=False):
    if deriv is False:
        exp = np.exp(x)
        denom = np.sum(np.exp(x), axis=1)
        denom = denom[:, np.newaxis]
        return exp / denom
    else:
        return 1


# Use Mean Square Error for error, but half MSE for derivative
# This allows a common error measure for output to user, but a simpler derivative
# This is fine because it only changes the derivative by a constant
def half_mse(x, y, deriv=False):
    if deriv is False:
        return (y - x) * (y - x)
    else:
        return -(y - x)


# Cross-Entropy Loss (Only works for 2d arrays for now)
# Returns the gradient of cross entropy wrt z_o
def cross_entropy(x, y, deriv=False):
    if deriv is False:
        return (-1 * np.sum(y * np.log(x), axis=1))[:, np.newaxis]
    else:
        return x - y
