import copy
import numpy as np

np.random.seed(0)


def rand(lower, upper):
    return (upper - lower) * np.random.random() + lower


def sigmoid(x):
    output = 1/(1+np.exp(-x))
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


def generate_case():
    dim = 8
    x = int(rand(0, 127))
    y = int(rand(0, 127))
    z = x + y
    x = bin(x)[2:]
    y = bin(y)[2:]
    z = bin(z)[2:]
    # align
    k = dim - len(x)
    if k > 0:
        x = "0" * k + x
    k = dim - len(y)
    if k > 0:
        y = "0" * k + y
    k = dim - len(z)
    if k > 0:
        z = "0" * k + z
        
    return (x, y), z


class LSTM:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_weights = []  # (input, hidden)
        self.output_weights = []  # (hidden, output)
        self.hidden_weights = []  # (hidden, hidden)
        self.input_updates = []  # (input, hidden)
        self.output_updates = []  # (hidden, output)
        self.hidden_updates = []  # (hidden, hidden)

    def setup(self, ni, nh, no):
        self.input_n = ni
        self.hidden_n = nh
        self.output_n = no
        self.input_weights = make_rand_mat(self.input_n, self.hidden_n)
        self.output_weights = make_rand_mat(self.hidden_n, self.output_n)
        self.hidden_weights = make_rand_mat(self.hidden_n, self.hidden_n)
        self.input_updates = make_mat(self.input_n, self.hidden_n, fill=0)
        self.output_updates = make_mat(self.hidden_n, self.output_n, fill=0)
        self.hidden_updates = make_mat(self.hidden_n, self.hidden_n, fill=0)

    def train(self):
        pass

    def predict(self, inputs):
        pass

    def train(self, cases, labels, dim, learn=0.1, limit=1000):
        for k in range(len(cases)):
            case = cases[k]
            label = labels[k]
            
            layer_2_deltas = []
            layer_1_values = [np.zeros(self.hidden_n)]

            error = 0.0
            guess = np.zeros_like(label)

            for i in range(dim):
                # todo
                X = np.array([[case[0][dim - i - 1], case[1][dim - i - 1]]])
                y = np.array([[label[dim - i - 1]]]).T

                layer_1 = sigmoid(np.dot(X, self.input_weights) + np.dot(layer_1_values[-1], self.hidden_weights))

                layer_2 = sigmoid(np.dot(layer_1, self.output_weights))

                layer_2_error = y - layer_2
                layer_2_deltas.append((layer_2_error) * sigmoid_derivative(layer_2))
                error += np.abs(layer_2_error[0])

                guess[dim - i - 1] = np.round(layer_2[0][0])

                # store hidden layer so we can use it in the next timestep
                layer_1_values.append(copy.deepcopy(layer_1))

            future_layer_1_delta = np.zeros(self.hidden_n)

            for i in range(dim):
                X = np.array([[case[0][i], case[1][i]]])
                layer_1 = layer_1_values[-i - 1]
                prev_layer_1 = layer_1_values[-i - 2]

                # error at output layer
                layer_2_delta = layer_2_deltas[-i - 1]
                # error at hidden layer
                layer_1_delta = (np.dot(future_layer_1_delta, self.hidden_weights.T) +
                                 layer_2_delta.dot(self.output_weights.T)) * sigmoid_derivative(layer_1)
                # let's update all our weights so we can try again
                self.output_updates += np.atleast_2d(layer_1).T.dot(layer_2_delta)
                self.hidden_updates += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
                self.input_updates += X.T.dot(layer_1_delta)

                future_layer_1_delta = layer_1_delta

            self.input_weights += self.input_updates * learn
            self.output_weights += self.output_updates * learn
            self.hidden_weights += self.hidden_updates * learn

            self.input_updates *= 0
            self.output_updates *= 0
            self.hidden_updates *= 0

            # print out progress
            if k % 1000 == 0:
                print "Error:" + str(error)
                print "Pred:" + str(guess)
                print "True:" + str(label)
                out = 0
                for index, x in enumerate(reversed(guess)):
                    out += x * pow(2, index)
                print str(case[0]) + " + " + str(case[1]) + " = " + str(out)
                print "------------"

    def test(self):
        self.setup(2, 16, 1)

        cases = []
        labels = []
        for i in range(1000):
            case, label = generate_case()
            cases.append(case)
            labels.append(label)

        self.train(cases, labels, 8)

if __name__ == '__main__':
    nn = LSTM()
    nn.test()
