import copy
import numpy as np


def rand(lower, upper):
    return (upper - lower) * np.random.random() + lower


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_derivative(y):
    return y * (1 - y)


def make_mat(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return np.array(mat)


def make_rand_mat(m, n, lower=-1, upper=1):
    mat = []
    for i in range(m):
        mat.append([rand(lower, upper)] * n)
    return np.array(mat)


def int_to_bin(x, dim=0):
    x = bin(x)[2:]
    # align
    k = dim - len(x)
    if k > 0:
        x = "0" * k + x
    return x


class LSTM:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_weights = []  # (input, hidden)
        self.output_weights = []  # (hidden, output)
        self.hidden_weights = []  # (hidden, hidden)

    def setup(self, ni, nh, no):
        self.input_n = ni
        self.hidden_n = nh
        self.output_n = no
        self.input_weights = make_rand_mat(self.input_n, self.hidden_n)
        self.output_weights = make_rand_mat(self.hidden_n, self.output_n)
        self.hidden_weights = make_rand_mat(self.hidden_n, self.hidden_n)

    def predict(self, inputs):
        pass

    def train(self, case=None, label=None, dim=0, learn=0.1):
        input_updates = np.zeros_like(self.input_weights)
        output_updates = np.zeros_like(self.output_weights)
        hidden_updates = np.zeros_like(self.hidden_weights)

        a = case[0]
        b = case[1]
        c = label

        d = np.zeros_like(c)

        error = 0

        layer_2_deltas = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.hidden_n))

        # moving along the positions in the binary encoding
        for position in range(dim):
            # generate input and output
            X = np.array([[a[dim - position - 1], b[dim - position - 1]]])
            y = np.array([[c[dim - position - 1]]]).T

            # hidden layer (input ~+ prev_hidden)
            layer_1 = sigmoid(np.dot(X, self.input_weights) + np.dot(layer_1_values[-1], self.hidden_weights))

            # output layer (new binary representation)
            layer_2 = sigmoid(np.dot(layer_1, self.output_weights))

            # did we miss?... if so by how much?
            layer_2_error = y - layer_2
            layer_2_deltas.append((layer_2_error) * sigmoid_derivative(layer_2))
            error += np.abs(layer_2_error[0])

            d[dim - position - 1] = np.round(layer_2[0][0])

            layer_1_values.append(copy.deepcopy(layer_1))

        future_layer_1_delta = np.zeros(self.hidden_n)

        for position in range(dim):
            X = np.array([[a[position], b[position]]])
            layer_1 = layer_1_values[-position - 1]
            prev_layer_1 = layer_1_values[-position - 2]

            # error at output layer
            layer_2_delta = layer_2_deltas[-position - 1]
            # error at hidden layer
            layer_1_delta = (future_layer_1_delta.dot(self.hidden_weights.T) +
                             layer_2_delta.dot(self.output_weights.T)) * sigmoid_derivative(layer_1)
            # let's update all our weights so we can try again
            output_updates += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            hidden_updates += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            input_updates += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

        self.input_weights += input_updates * learn
        self.output_weights += output_updates * learn
        self.hidden_weights += hidden_updates * learn

        return d, error

    def test(self):
        self.setup(2, 16, 1)
        for i in range(20000):
            # generate a simple addition problem (a + b = c)
            a_int = int(rand(0, 127))
            a = int_to_bin(a_int, dim=8)
            a = np.array([int(t) for t in a])

            b_int = int(rand(0, 127))
            b = int_to_bin(b_int, dim=8)
            b = np.array([int(t) for t in b])

            # true answer
            c_int = a_int + b_int
            c = int_to_bin(c_int, dim=8)
            c = np.array([int(t) for t in c])

            d, error = self.train([a, b], c, dim=8)

            if i % 1000 == 0:
                print "Error:" + str(error)
                print "Pred:" + str(d)
                print "True:" + str(c)
                out = 0
                for index, x in enumerate(reversed(d)):
                    out += x * pow(2, index)
                print str(a_int) + " + " + str(b_int) + " = " + str(out)
                print "------------"

if __name__ == '__main__':
    nn = LSTM()
    nn.test()
