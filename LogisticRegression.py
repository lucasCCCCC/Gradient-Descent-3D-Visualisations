import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
from scipy.special import expit
from random import randint


class LogisticRegression:

    def __init__(self, w0, w1, learningRate):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate

    def getWeights(self):
        return self.w1, self.w0

    def sigmoid(self, X):
        Zs = self.w1 * X + self.w0
        return 1 / (1 + np.exp(-Zs))

    def computeCrossEntropyLoss(self, X, Y):
        return - np.mean(Y * np.log(self.sigmoid(X))
                         + (1 - Y) * np.log(1 - self.sigmoid(X)))

    def gradientDescent(self, X, Y):
        y_val = expit(self.w1 * X + self.w0)

        w0_pd = np.mean(y_val - Y)
        w1_pd = np.mean((y_val - Y) * X)

        self.w0 = self.w0 - self.learningRate * w0_pd
        self.w1 = self.w1 - self.learningRate * w1_pd


test_data = np.array([[1, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [4, 0], [4.5, 0],
                      [5.5, 0], [6, 1], [7, 1], [7.5, 1], [8, 1], [8.5, 1], [9, 1]])


x_train_data = test_data[:, :1]
y_train_data = test_data[:, 1:]

model = LogisticRegression(randint(-10, 10), randint(-10, 10), 0.1)

w0Iterations = []
w1Iterations = []
predictedYValues = []


for i in range(5000):
    model.gradientDescent(x_train_data, y_train_data)
    w0Iterations.append(model.getWeights()[1])
    w1Iterations.append(model.getWeights()[0])
    predictedYValues.append(model.sigmoid(x_train_data))
    # print(model.computeCrossEntropyLoss(x_train_data, y_train_data))

# print("Final Model: y = 1/1 + exp(-(", model.getWeights()[0], "x + ", model.getWeights()[1], "))")


fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1, 2, 2)
ax1.set_title("Regression Visualisation")

ax2 = fig.add_subplot(1, 2, 1, projection='3d')
ax2.set_xlabel("w1")
ax2.set_ylabel("w0")
ax2.set_zlabel("(Cross Entropy) Log Loss")
ax2.set_title("Gradient Descent Visualisation")


def computePairCrossEntropyLoss(xs, ys, w1, w0):
    Z = expit(xs @ w1.T + w0)
    return - np.mean(ys * np.log(Z) - (1 - ys) * np.log(1 - Z))


x = np.linspace(-20, 2.9, 40)
y = np.linspace(-10, 10, 40)
X_3d, Y_3d = np.meshgrid(x, y)

z = np.array([computePairCrossEntropyLoss(x_train_data, y_train_data, np.array([[xs]]), np.array([[ys]])) for
              xs, ys in zip(np.ravel(X_3d), np.ravel(Y_3d))])

Z_3d = z.reshape(X_3d.shape)

camera = Camera(fig)

for i in range(0, len(w0Iterations), 500):

    ax1.scatter(x_train_data, y_train_data, color="blue")
    ax1.plot(x_train_data, predictedYValues[i], color="red")

    ax1.legend(['Iteration: {:d}\nModel: y=1/1+exp(-({:.4f}x+{:.4f}))'.format(i, float(w1Iterations[i]),
                                                                              float(w0Iterations[i]))],
                                                                              bbox_to_anchor=(0, 0))


    ax2.plot_surface(X_3d, Y_3d, Z_3d, rstride=1, cstride=1, cmap="Reds", alpha=0.7)
    ax2.scatter(w1Iterations[i], w0Iterations[i], predictedYValues[i], antialiased=False, color="black")

    camera.snap()

animation = camera.animate(interval=1, repeat=False, repeat_delay=0)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()
