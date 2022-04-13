import numpy as np


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

