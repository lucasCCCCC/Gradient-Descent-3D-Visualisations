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
        y_val = self.w1 * X + self.w0

        w0_pd = -(2 / len(X)) * sum(Y - y_val)
        w1_pd = -(2 / len(X)) * sum(X * (Y - y_val))

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
ax1 = fig.add_subplot(1, 2, 2)
ax2 = fig.add_subplot(1, 2, 1, projection='3d')
ax2.view_init(elev=20, azim=31)


def pairCosts(xs, ys, w0, w1):
    return np.mean((ys - (xs * w1 + w0)) ** 2)


x = np.linspace(-10, 10, 10)
y = np.linspace(-10, 10, 10)
X_3d, Y_3d = np.meshgrid(x, y)

z = np.array([pairCosts(x_train_data, y_train_data, np.array([[xs]]), np.array([[ys]]))
              for xs, ys in zip(np.ravel(X_3d), np.ravel(Y_3d))])

Z_3d = z.reshape(X_3d.shape)

camera = Camera(fig)
fig.suptitle("Linear Regression Visualiser")

for i in range(0, len(w0Iterations), 5):
    ax1.scatter(x_train_data, y_train_data, color="blue")
    ax1.plot(x_train_data, predictedYValues[i], color="red")
    ax1.legend(['Iteration: {:d}\nModel: y={:.4f}x+{:.4f}'.format(i, float(w1Iterations[i]), float(w0Iterations[i]))],
               loc='lower right')

    ax2.plot_surface(X_3d, Y_3d, Z_3d, rstride=2, cstride=2, cmap="jet", alpha=0.8)
    ax2.set_xlabel("w0")
    ax2.set_ylabel("w1")
    ax2.set_zlabel("L2 Loss")

    camera.snap()

animation = camera.animate(interval=100, repeat=False)

plt.show()
