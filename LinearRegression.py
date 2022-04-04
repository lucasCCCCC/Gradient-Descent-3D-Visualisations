import numpy as np


class LinearRegression:

    def __init__(self, w0, w1, learningRate):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate

    def getWeights(self):
        return self.w1, self.w0

    def computeL2Loss(self, X, Y):
        return (1 / len(X)) * sum([val ** 2 for val in (self.w1 * X + self.w0 - Y)])

    def gradientDescent(self, X, Y):
        y = self.w1 * X + self.w0

        w0_pd = -(2 / len(X)) * sum(Y - y)
        w1_pd = -(2 / len(X)) * sum(X * (Y - y))

        self.w0 = self.w0 - self.learningRate * w0_pd
        self.w1 = self.w1 - self.learningRate * w1_pd


test_data = np.array([[1, 1], [1, 2], [3, 5], [3, 8], [2, 2], [2, 3], [5, 5], [4, 5], [4, 7], [4, 8],
                      [5, 3], [6, 5], [6, 6], [6, 8], [9, 9], [9, 11], [10, 9], [10, 13], [13, 8], [13, 15], [14, 3],
                      [16, 16], [16, 15], [17, 20], [18, 19], [23, 20], [24, 26], [25, 18], [20, 19], [23, 23],
                      [22, 24], [26, 24]])

x_train_data = test_data[:, :1]
y_train_data = test_data[:, 1:]

model = LinearRegression(1, 1, 0.001)
print("Training model")

for i in range(1000):
    model.gradientDescent(x_train_data, y_train_data)
    print("Current Model: ", "y = ", model.getWeights()[1], "x + ", model.getWeights()[0])
    print("Loss: ", model.computeL2Loss(x_train_data, y_train_data))
    print("Iteration: ", i)

print("Final Model: ", "y = ", model.getWeights()[1], "x + ", model.getWeights()[0])
