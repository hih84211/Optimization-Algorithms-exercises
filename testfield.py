# https://medium.com/edureka/linear-regression-in-python-e66f869cb6ce
# Importing Necessary Libraries
import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)


def f(t):
    return 2 * (t ** 2) + 3 * t + 4 + 50 * (random.random() - 0.5)


l = list([[],[]])
for i in range(1, 21):
    l[0].append(i)
    l[1].append(f(i))

"""
# Load and plot data files
"""
# Load the training data hw1xtr.dat and hw1ytr.dat into the memory
#train_features = pd.read_csv('xtr.dat',  header = None)
#train_desired_outputs = pd.read_csv('ytr.dat', header = None)

# # Plot training_data and desired_outputs
train_features_vals = np.array(l[0])
train_desired_outputs_vals = np.array(l[1])
plt.scatter(train_features_vals, train_desired_outputs_vals, color = 'g', marker = 'o', s = 30)
plt.title('Training Data')
plt.show()

# Load the test data hw1xte.dat and hw1yte.dat into the memory
#test_features = pd.read_csv('xte.dat',  header = None)
#test_desired_outputs = pd.read_csv('yte.dat', header = None)
# # Plot training_data and desired_outputs
"""plt.scatter(test_features.values, test_desired_outputs.values, color = 'b', marker = 'o', s = 30)
plt.title('Testing Data')
plt.show()
# Train linear regression model on training set
"""

N = len(l[0])
X = np.c_[np.ones(N), train_features_vals, np.square(l[0])]
A = np.linalg.inv(X.T@X)
D = A@X.T
result = D@train_desired_outputs_vals

y_pred = []


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    x_line = np.linspace(train_features_vals.min(), train_features_vals.max(), 100)
    global y_pred
    y_pred = b[2] * np.square(x_line) + b[1] * x_line + b[0]
    regression_line = y_pred
    # Plotting the regression line
    plt.plot(x_line, regression_line, color="g")
    # Putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    # Plot scatter plot


plt.scatter(train_features_vals, train_desired_outputs_vals, color='m', marker='o', s=30)
plot_regression_line(train_features_vals, train_desired_outputs_vals, result)
plt.show()

# Find average error on the training set
A = np.square(result[2] * np.square(train_features_vals) + result[1] * train_features_vals +
              result[0] - train_desired_outputs_vals)
error = np.sum(A) / N
print('Average error on the training set: ', error)
