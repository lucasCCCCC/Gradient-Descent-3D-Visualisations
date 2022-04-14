import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera


class LogisticRegression:

    def __init__(self, w0, w1, learningRate):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate

    def getWeights(self):
        return self.w1, self.w0

    def sigmoid(self, X):
        Z = self.w1 * X + self.w0
        return 1 / (1 + np.exp(-Z))

    def computeCrossEntropyLoss(self, X, Y):
        pass

    def gradientDescent(self, X, Y):
        y_val = self.sigmoid(X)

        w0_pd = np.mean(y_val - Y)
        w1_pd = np.mean((y_val - Y)*X)

        self.w0 = self.w0 - self.learningRate * w0_pd
        self.w1 = self.w1 - self.learningRate * w1_pd


test_data = np.array([[1, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [3.5, 0], [4, 0],
                      [6, 1], [6.5, 1], [7, 1], [7.5, 1], [8, 1], [8.5, 1], [9, 1]])

x_train_data = test_data[:, :1]
y_train_data = test_data[:, 1:]

model = LogisticRegression(20, 10, 0.1)

w0Iterations = []
w1Iterations = []
predictedYValues = []


for i in range(30000):
    model.gradientDescent(x_train_data, y_train_data)
    w0Iterations.append(model.getWeights()[1])
    w1Iterations.append(model.getWeights()[0])
    predictedYValues.append(model.sigmoid(x_train_data))


print("Final Model: y = 1/1 + exp(-(", model.getWeights()[0], "x + ", model.getWeights()[1], "))")


fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot()
ax1.set_title("Visualising sigmoid function learning the best fit")

camera = Camera(fig)

for i in range(0, len(w0Iterations), 50):
    ax1.scatter(x_train_data, y_train_data, color="blue")
    # need to smooth out the sigmoid plot
    ax1.plot(x_train_data, predictedYValues[i], color="red")
    camera.snap()

animation = camera.animate(interval=3)

plt.show()
