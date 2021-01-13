import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


'''
train_features_vals = np.array(l[0])
train_desired_outputs_vals = np.array(l[1])
plt.scatter(train_features_vals, train_desired_outputs_vals, color = 'g', marker = 'o', s = 30)
plt.title('Training Data')
plt.show()
'''

"""plt.scatter(test_features.values, test_desired_outputs.values, color = 'b', marker = 'o', s = 30)
plt.title('Testing Data')
plt.show()
# Train linear regression model on training set
"""
class ls():
    def __init__(self, data_train, data_test=None):
        self.data_train = data_train
        self.data_test = data_test
        self.x_train = np.array(data_train[0])
        self.y_train = np.array(data_train[1])

        if data_test:
            self.x_test = data_test[0]
            self.y_test = data_test[1]


    def train(self):
        N = len(self.x_train)
        X = np.c_[np.ones(N), self.x_train, np.square(self.x_train)]
        A = np.linalg.inv(X.T @ X)
        D = A @ X.T
        self.result = D @ self.y_train

    def _get_error(self, z):
        x = np.array(z[0])
        y = np.array(z[1])
        A = np.square(self.result[2] * np.square(x) + self.result[1] * x +
                      self.result[0] - y)
        N = len(z)
        error = np.sum(A) / N

        return error

    def plot_regression_line(self, z, plot_title):
        plt.title(plot_title)
        plt.scatter(z[0], z[1], color="m",
                    marker="o", s=30)

        x_line = np.linspace(self.x_train.min(), self.y_train.max(), 100)
        global y_pred
        y_pred = self.result[2] * np.square(x_line) + self.result[1] * x_line + self.result[0]
        regression_line = y_pred
        plt.plot(x_line, regression_line, color="g")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        error = self._get_error(z)

        return error



if __name__ == '__main__':

    def f(t):
        return 2 * (t ** 2) + 3 * t + 4 + 50 * (random.random() - 0.5)

    data1 = []
    data2 = []
    for i in range(1, 21):
        data1.append(i)
        data2.append(f(i))
    data_set = [data1, data2]

    print('--------------Problem 6(a)----------------')
    ls_a = ls(data_set)
    ls_a.train()
    error = ls_a.plot_regression_line(data_set, 'Training data')
    print('Average error of the whole data pairs set: ', error)


