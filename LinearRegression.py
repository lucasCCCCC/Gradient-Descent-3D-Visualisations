from matplotlib import pyplot as plt
import numpy as np
from celluloid import Camera
from random import randint


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

model = LinearRegression(randint(-10, 10), randint(-10, 10), 0.001)

#print("Training model")

for i in range(1000):
    model.gradientDescent(x_train_data, y_train_data)
    w0Iterations.append(model.getWeights()[1])
    w1Iterations.append(model.getWeights()[0])
    costIterations.append(model.computeL2Loss(x_train_data, y_train_data))
    predictedYValues.append(model.computeY(x_train_data))

    #print("Current Model: ", "y = ", model.getWeights()[1], "x + ", model.getWeights()[0])
    #print("Loss: ", model.computeL2Loss(x_train_data, y_train_data))
    #print("Iteration: ", i)

#print("Final Model: ", "y = ", model.getWeights()[1], "x + ", model.getWeights()[0])

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1, 2, 2)
ax1.set_title("Regression Visualisation")

ax2 = fig.add_subplot(1, 2, 1, projection='3d')
ax2.view_init(45, -45)
ax2.set_xlabel("w1")
ax2.set_ylabel("w0")
ax2.set_zlabel("L2 Loss")
ax2.set_title("Gradient Descent Visualisation")


def pairCosts(xs, ys, w0, w1):
    return np.mean((ys - (xs * w1 + w0)) ** 2)


x = np.linspace(-10, 10, 5)
y = np.linspace(-10, 10, 5)
X_3d, Y_3d = np.meshgrid(x, y)

z = np.array([pairCosts(x_train_data, y_train_data, np.array([[xs]]), np.array([[ys]]))
              for xs, ys in zip(np.ravel(X_3d), np.ravel(Y_3d))])

Z_3d = z.reshape(X_3d.shape)

camera = Camera(fig)
fig.suptitle("Linear Regression Visualiser")

for i in range(0, len(w0Iterations), 10):
    ax1.scatter(x_train_data, y_train_data, color="blue")
    ax1.plot(x_train_data, predictedYValues[i], color="red")
    ax1.legend(['Iteration: {:d}\nModel: y={:.4f}x+{:.4f}'.format(i, float(w1Iterations[i]), float(w0Iterations[i]))],
               loc='lower right')

    ax2.plot_surface(X_3d, Y_3d, Z_3d, rstride=1, cstride=1, cmap="Oranges", alpha=0.7)
    ax2.scatter(w1Iterations[i], w0Iterations[i], predictedYValues[i], color="black")

    camera.snap()

animation = camera.animate(interval=5, repeat=False, repeat_delay=0)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()
