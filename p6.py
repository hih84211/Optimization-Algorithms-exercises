import numpy as np
from math import *
import random
# 未完成
# https://github.com/jhumphry/regressions
# https://github.com/Networks-Learning/l1-ls.py/blob/master/l1ls/l1_ls.py
# https://github.com/craig-m-k/Recursive-least-squares
class CLS():

    """Classical Least Squares Regression
    The classical least squares regression approach is to initially swap the
    roles of the X and Y variables, perform linear regression and then to
    invert the result. It is useful when the number of X variables is larger
    than the number of calibration samples available, when conventional
    multiple linear regression would be unable to proceed.
    Note :
        The regression matrix A_pinv is found using the pseudo-inverse. In
        order for this to be calculable, the number of calibration samples
        ``N`` has be be larger than the number of Y variables ``m``, the
        number of X variables ``n`` must at least equal the number of Y
        variables, there must not be any collinearities in the calibration Y
        data and Yt X must be non-singular.
    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
    Attributes:
        A (ndarray m x n): Resulting regression matrix of X on Y
        A_pinv (ndarray m x n): Pseudo-inverse of A
    """

    def __init__(self, X, Y):
        if X.shape[1] < Yc.shape[1]:
            raise ParameterError('CLS requires at least as input variables '
                                 '(columns of X data) as output variables '
                                 '(columns of Y data)')

        self.A = linalg.inv(Yc.T @ Yc) @ Yc.T @ Xc
        self.A_pinv = self.A.T @ linalg.inv(self.A @ self.A.T)

    def prediction(self, Z):

        """Predict the output resulting from a given input
        Args:
            Z (ndarray of floats): The input on which to make the
                prediction. Must either be a one dimensional array of the
                same length as the number of calibration X variables, or a
                two dimensional array with the same number of columns as
                the calibration X data and one row for each input row.
        Returns:
            Y (ndarray of floats) : The predicted output - either a one
            dimensional array of the same length as the number of
            calibration Y variables or a two dimensional array with the
            same number of columns as the calibration Y data and one row
            for each input row.
        """

        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the same '
                                     'number of variables as the original X '
                                     'data')
            return self.Y_offset + (Z - self.X_offset) @ self.A_pinv
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            for i in range(0, Z.shape[0]):
                result[i, :] = self.Y_offset + (Z[i, :] - self.X_offset) \
                    @ self.A_pinv
            return result
class RLS:
    def __init__(self, num_vars, lam, delta):
        '''
        num_vars: number of variables including constant
        lam: forgetting factor, usually very close to 1.
        '''
        self.num_vars = num_vars

        # delta controls the initial state.
        self.A = delta * np.matrix(np.identity(self.num_vars))
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1], 1)

        # Variables needed for add_obs
        self.lam_inv = lam ** (-1)
        self.sqrt_lam_inv = sqrt(self.lam_inv)

        # A priori error
        self.a_priori_error = 0

        # Count of number of observations added
        self.num_obs = 0

    def add_obs(self, x, t):
        '''
        Add the observation x with label t.
        x is a column vector as a numpy matrix
        t is a real scalar
        '''
        z = self.lam_inv * self.A * x
        alpha = float((1 + x.T * z) ** (-1))
        self.a_priori_error = float(t - self.w.T * x)
        self.w = self.w + (t - alpha * float(x.T * (self.w + t * z))) * z
        self.A -= alpha * z * z.T
        self.num_obs += 1

    def fit(self, X, y):
        '''
        Fit a model to X,y.
        X and y are numpy arrays.
        Individual observations in X should have a prepended 1 for constant coefficient.
        '''
        for i in range(len(X)):
            x = np.transpose(np.matrix(X[i]))
            self.add_obs(x, y[i])

    def get_error(self):
        '''
        Finds the a priori (instantaneous) error.
        Does not calculate the cumulative effect
        of round-off errors.
        '''
        return self.a_priori_error

    def predict(self, x):
        '''
        Predict the value of observation x. x should be a numpy matrix (col vector)
        '''
        return float(self.w.T * x)

    if __name__ == '__main__':
        def f(t):
            return 2*(t**2) + 3*t + 4 + 50*(random.random() - 0.5)

        l = list()
        for i in range(1, 21):
            l.append((i, f(i)))

        print(l)
