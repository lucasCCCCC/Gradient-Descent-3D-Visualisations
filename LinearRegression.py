from matplotlib import pyplot as plt
import numpy as np
from celluloid import Camera


class LinearRegression:

    def __init__(self, w0, w1, learningRate):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate

    def getWeights(self):
        return self.w1, self.w0

    def computeY(self, X):
        return self.w1 * X + self.w0

    def computeL2Loss(self, X, Y):
        return (1 / len(X)) * sum([val ** 2 for val in (self.w1 * X + self.w0 - Y)])

    def gradientDescent(self, X, Y):
        y = self.w1 * X + self.w0

        w0_pd = -(2 / len(X)) * sum(Y - y)
        w1_pd = -(2 / len(X)) * sum(X * (Y - y))

        self.w0 = self.w0 - self.learningRate * w0_pd
        self.w1 = self.w1 - self.learningRate * w1_pd


test_data = np.array([[1, 1], [2, 2], [3, 4], [4, 4], [5, 5], [5, 6], [6, 5], [7, 7], [7, 6], [8, 8], [9, 7], [10, 11]])

x_train_data = test_data[:, :1]
y_train_data = test_data[:, 1:]
w0Iterations = []
w1Iterations = []
costIterations = []
predictedYValues = []


model = LinearRegression(2, -3, 0.001)

print("Training model")

for i in range(1000):
    model.gradientDescent(x_train_data, y_train_data)
    w0Iterations.append(model.getWeights()[1])
    w1Iterations.append(model.getWeights()[0])
    costIterations.append(model.computeL2Loss(x_train_data, y_train_data))
    predictedYValues.append(model.computeY(x_train_data))

    print("Current Model: ", "y = ", model.getWeights()[1], "x + ", model.getWeights()[0])
    print("Loss: ", model.computeL2Loss(x_train_data, y_train_data))
    print("Iteration: ", i)

print("Final Model: ", "y = ", model.getWeights()[1], "x + ", model.getWeights()[0])

fig = plt.figure()
ax1 = fig.add_subplot()


camera = Camera(fig)
fig.suptitle("Regression Line & Gradient Descent Visualisation")

for i in range(0, len(w0Iterations), 5):

    ax1.scatter(x_train_data, y_train_data, color="blue")
    ax1.plot(x_train_data, predictedYValues[i], color="red")
    ax1.legend(['Iteration: {:f}\nModel: y={:.5f}x+{:.5f}'.format(i, float(w1Iterations[i]), float(w0Iterations[i]))],
               loc='lower right')

    camera.snap()

animation = camera.animate(interval=100, repeat=False)

plt.show()
