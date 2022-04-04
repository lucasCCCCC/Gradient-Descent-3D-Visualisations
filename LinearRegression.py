import numpy as np


def compute_L2_loss(X, Y, w1, w0):
    cost = 0
    for i in range(len(X)):
        cost = (1 / len(X)) * sum([val ** 2 for val in (w1 * X + w0 - Y)])
    return cost


def gradient_descent(X, Y, w0, w1, learning_rate):
    iterations = 1000
    data_set_size = len(X)

    print("Training model")

    for i in range(iterations):
        y = w1 * X + w0
        w0_pd = -(2 / data_set_size) * sum(Y - y)
        w1_pd = -(2 / data_set_size) * sum(X * (Y - y))
        w0 = w0 - learning_rate * w0_pd
        w1 = w1 - learning_rate * w1_pd

        print("Current model: y = ", w1, "x + ", w0)
        print("Loss : ", compute_L2_loss(X, Y, w1, w0))
        print("Iteration: ", i)

    print("Final model: y = ", w1, "x + ", w0)


test_data = np.array([[1, 1], [1, 2], [3, 5], [3, 8], [2, 2], [2, 3], [5, 5], [4, 5], [4, 7], [4, 8],
                      [5, 3], [6, 5], [6, 6], [6, 8], [9, 9], [9, 11], [10, 9], [10, 13], [13, 8], [13, 15]
                         , [14, 3], [16, 16], [16, 15], [17, 20], [18, 19], [23, 20], [24, 26], [25, 18]
                         , [20, 19], [23, 23], [22, 24], [26, 24]])

X = test_data[:, :1]
Y = test_data[:, 1:]

gradient_descent(X, Y, 1, 1, 0.001)
