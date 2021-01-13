import random

import numpy as np
import sympy as sp
from sympy import ordered, Matrix

DFP = 1
BFGS = 2


def Gradient(f, X=None):
    v = list(ordered(f.free_symbols))
    grd = lambda fn, vn: Matrix([fn]).jacobian(vn)
    if X is not None:
        return list(grd(f, v).subs(list(zip(v, X))))
    else:
        return grd(f, v)


def secant_search(g, X, d, lower=-10, upper=10, epsilon=0.00001):
    v = list(ordered(g.free_symbols))
    max = 500
    alpha_curr = 0
    alpha = epsilon

    dphi_zero = np.dot(np.array(list(g.subs(list(zip(v, X)))), dtype=np.float), d)
    dphi_curr = dphi_zero

    i = 0
    while abs(dphi_curr) > epsilon * abs(dphi_zero):
        alpha_old = alpha_curr
        alpha_curr = alpha
        dphi_old = dphi_curr
        dphi_curr = np.dot(np.array(list(g.subs(list(zip(v, X + (alpha_curr * d))))), dtype=np.float), d)
        alpha = (dphi_curr * alpha_old - dphi_old * alpha_curr) / (dphi_curr - dphi_old)
        i += 1
        if i % 2 == 0:
            print("i={}, alpha_curr={}, alpha_old={}, alpha={}".format(i, alpha_curr, alpha_old, alpha))
        if (i >= max) or (abs(dphi_curr) > epsilon * abs(dphi_zero)):
            return alpha

    def bfgs(self, f, x0, d0, g0, Q0, epslon, i, alpha):
        '''
            Broyden-Fletcher-Goldfarb-Shanno
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        g = Gradient(f, x0)
        if sum(abs(d0)) < epslon or i is not 0:
            Q = [self.params['hessian']['initial'] if self.params['hessian']['initial'] else np.identity(len(x0))][0]
        else:
            q = (g - g0)[np.newaxis].T
            p = (alpha * d0)[np.newaxis].T
            Q = Q0 + (1.0 + q.T.dot(Q0).dot(q) / (q.T.dot(p))) * (p.dot(p.T)) / (p.T.dot(q)) - (
                        p.dot(q.T).dot(Q0) + Q0.dot(q).dot(p.T)) / (q.T.dot(p))
        d = -Q.dot(g)
        return d, g, Q

def QuasiNewton(f, max_iter, x0, epsilon):
    i = 0
    xk = x0
    norm_values = []

    a = np.random.random_integers(-20, 20+1, size=(2, 2))
    b = a.T
    hk = (a + b) / 2
    grad_f = Gradient(f)

    while i < max_iter:
        norm = np.linalg.norm(np.array(Gradient(f, xk), dtype=np.float))
        if norm < epsilon:
            break
        else:
            B_inv = np.linalg.inv(B[i])

            # print(grad)
            p = -np.dot(grad, B_inv)
            # print(p)
            alpha = secant_search(grad_f, x_values[i], p)
            x_values[i + 1] = x_values[i] + alpha * p

            del_k = np.array(x_values[i + 1] - x_values[i])

            #gamma_k = np.dot(del_k, grad)
            gamma_k = del_k * grad

            B[i + 1] = B[i] + np.dot(gamma_k.T, gamma_k) / np.dot(del_k, gamma_k) - \
                       np.dot(np.dot(gamma_k.T, del_k.T), np.dot(del_k, gamma_k))/np.dot(np.dot(del_k, gamma_k), del_k.T)

            norm_values.append(norm)
        i += 1
    return (x_values[i-1], norm_values[i-1])


if __name__ == '__main__':
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    k, m, n = sp.symbols('k m n', integer=True)
    f, g, h = sp.symbols('f g h', cls=sp.Function)
    sp.init_printing(use_unicode=True)
    #f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    f = (x1 - 1) ** 2 + (2 - x2 ** 2) ** 2 + 4  # * (x3 - 3)**4
    v = list(ordered(f.free_symbols))

    print(QuasiNewton(f, 10000, [-2, 2], 0.0001))
