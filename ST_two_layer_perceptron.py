from functools import reduce

from forward_mode import *
from random import uniform


def vplus(u, v):
    return [ad_plus(u[i], v[i]) for i in range(0, len(u))]


def ktimesv(k, u):
    return [ad_times(k, u[i]) for i in range(0, len(u))]


def dot(u, v):
    sum = 0
    for i in range(0, len(u)):
        sum = ad_plus(sum, ad_times(u[i], v[i]))
    return sum


def mtimesv(m, v):
    return [dot(m[i], v) for i in range(0, len(m))]


def vminus(u, v):
    return vplus(u, ktimesv(-1, v))


def distance(u, v):
    return dot(vminus(u, v), vminus(u, v))


def naive_gradient_descent(f, x0, learning_rate, n):
    x = x0
    for i in range(0, n):
        x = vminus(x, ktimesv(learning_rate, gradient(f)(x)))
    return x


def st_layer(point, theta):
    a_matrix = []
    a_matrix.append([theta[0], theta[1]])
    a_matrix.append([theta[3], theta[4]])
    b_const = [theta[2], theta[5]]
    return vplus(mtimesv(a_matrix, point), b_const)


def fc_layer(point, weights, biases):
    return vplus(mtimesv(weights, point), biases)


def sigmoid(x):
    return ad_divide(1, ad_plus(ad_exp(ad_minus(0, x)), 1))


def sigmoid_layer(point):
    return [sigmoid(point[i]) for i in range(0, len(point))]


def two_layer_perceptron(point, weights1, biases1, weights2, biases2, theta):
    hidden = sigmoid_layer(fc_layer(st_layer(point, theta), weights1, biases1))
    return fc_layer(hidden, weights2, biases2)


def cost(points, labels, weights1, biases1, weights2, biases2, theta):
    return reduce(ad_plus,
                  [distance(two_layer_perceptron(points[i],
                                                 weights1,
                                                 biases1,
                                                 weights2,
                                                 biases2,
                                                 theta),
                            [labels[i]])
                   for i in range(0, len(points))],
                  0)


def pack(weights1, biases1, weights2, biases2, theta):
    parameters = []
    for bias in biases1:
        parameters.append(bias)
    for row in weights1:
        for weight in row:
            parameters.append(weight)
    for bias in biases2:
        parameters.append(bias)
    for row in weights2:
        for weight in row:
            parameters.append(weight)
    for theta0 in theta:
        parameters.append(theta0)
    return parameters


def unpack(parameters, number_of_inputs, number_of_hidden):
    k = 0
    biases1 = []
    for j in range(0, number_of_hidden):
        biases1.append(parameters[k])
        k = k+1
    weights1 = []
    for j in range(0, number_of_hidden):
        row = []
        for i in range(0, number_of_inputs):
            row.append(parameters[k])
            k = k+1
        weights1.append(row)

    biases2 = []
    biases2.append(parameters[k])
    k = k+1
    row = []
    for i in range(0, number_of_hidden):
        row.append(parameters[k])
        k = k+1
    weights2 = [row]

    theta = []
    for i in range(0, 6):
        theta.append(parameters[k])
        k = k+1

    return weights1, biases1, weights2, biases2, theta


def initialize(points, labels, number_of_hidden):
    number_of_inputs = len(points[0])
    weights1 = [[uniform(-1, 1) for i in range(0, number_of_inputs)]
                for j in range(0, number_of_hidden)]
    biases1 = [uniform(-1, 1) for j in range(0, number_of_hidden)]
    weights2 = [[uniform(-1, 1) for j in range(0, number_of_hidden)]]
    biases2 = [uniform(-1, 1)]
    theta = [uniform(-1, 1) for j in range(0, 6)]
    return weights1, biases1, weights2, biases2, theta


def step(points, labels, weights1, biases1, weights2, biases2, theta):
    number_of_inputs = len(points[0])
    number_of_hidden = len(biases1)

    def loss(parameters):
        weights1, biases1, weights2, biases2, theta = unpack(
            parameters, number_of_inputs, number_of_hidden)
        return cost(points, labels, weights1, biases1, weights2, biases2, theta)
    parameters = pack(weights1, biases1, weights2, biases2, theta)
    parameters = vminus(parameters, ktimesv(0.01, gradient(loss)(parameters)))
    weights1, biases1, weights2, biases2, theta = unpack(
        parameters, number_of_inputs, number_of_hidden)
    return weights1, biases1, weights2, biases2, theta


def train(points, labels, number_of_hidden):
    number_of_inputs = len(points[0])

    def loss(parameters):
        weights1, biases1, weights2, biases2, theta = unpack(
            parameters, number_of_inputs, number_of_hidden)
        return cost(points, labels, weights1, biases1, weights2, biases2, theta)
    weights1 = [[uniform(-1, 1) for i in range(0, number_of_inputs)]
                for j in range(0, number_of_hidden)]
    biases1 = [uniform(-1, 1) for j in range(0, number_of_hidden)]
    weights2 = [[uniform(-1, 1) for j in range(0, number_of_hidden)]]
    biases2 = [uniform(-1, 1)]
    theta = [uniform(-1, 1) for j in range(0, 6)]
    parameters = pack(weights1, biases1, weights2, biases2, theta)
    # parameters = naive_gradient_descent(loss, parameters, 0.02, 5000)

    n_train = 0
    while cost(points, labels, weights1, biases1, weights2, biases2, theta) > 0.01 and n_train < 5000:
        parameters = naive_gradient_descent(loss, parameters, 0.1, 1)
        weights1, biases1, weights2, biases2, theta = unpack(
            parameters, number_of_inputs, number_of_hidden)
        n_train = n_train + 1
    return weights1, biases1, weights2, biases2, theta, n_train


def classify(point, weights1, biases1, weights2, biases2, theta):
    if two_layer_perceptron(point, weights1, biases1, weights2, biases2, theta)[0]<0:
        return -1
    else:
        return +1


def all_labels(labels):
    red = False
    blue = False
    for label in labels:
        if label < 0:
            red = True
        else:
            blue = True
    return red and blue
