import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D
from sympy import ordered, Matrix
from math import *

Hestenes_Stiefel = 1
Polak_Ribiere = 2
Fletcher_Reeves = 3

def golden_section_searcher(f, X, d, prev_val, lower, upper, epsilon):
    phi = (1.0 + sqrt(5.0)) / 2.0
    x1 = upper - ((phi - 1) * (upper - lower))
    x2 = lower + ((phi - 1) * (upper - lower))
    val = x1

    param2 = X - np.dot(x2, d)
    param2 = param2.tolist()

    param1 = X - np.dot(x1, d)
    param1 = param1.tolist()
    f1 = f.subs(list(zip(v, param1)))
    f2 = f.subs(list(zip(v, param2)))

    if f2 < f1:
        if x1 > x2:
            upper = x1
        else:
            lower = x1

    else:
        if x2 > x1:
            upper = x2
        else:
            lower = x2

    if abs(prev_val - val) <= epsilon:
        return val
    else:
        return golden_section_searcher(f, X, d, val, lower, upper, epsilon)

def secant_search(g, X, d, lower=-10, upper=10, epsilon=0.00001):
    v = list(ordered(g.free_symbols))
    max = 500
    alpha_curr = 0
    alpha = epsilon

    dphi_zero = np.dot(np.array(list(g.subs(list(zip(v, X)))), dtype=np.float), d)
    dphi_curr = dphi_zero

    i = 0
    while abs(dphi_curr) > epsilon*abs(dphi_zero):
        alpha_old = alpha_curr
        alpha_curr = alpha
        dphi_old = dphi_curr
        dphi_curr = np.dot(np.array(list(g.subs(list(zip(v, X+(alpha_curr*d))))), dtype=np.float), d)
        alpha = (dphi_curr * alpha_old - dphi_old * alpha_curr) / (dphi_curr - dphi_old)
        i += 1
        if i%2 == 0:
            print("i={}, alpha_curr={}, alpha_old={}, alpha={}".format(i, alpha_curr, alpha_old, alpha))
        if (i >= max) or (abs(dphi_curr) > epsilon * abs(dphi_zero)):
            #print('alpha: ', alpha)
            return alpha

def fibonacci_num(n):
    if n <= 2:
        return 1
    else:
        return fibonacci_num(n - 1) + fibonacci_num(n - 2)


def fibonacci_search(f, X, d, lower, upper, r):
    v = list(ordered(f.free_symbols))
    N = 0
    while fibonacci_num(N) < ((1 + 2 * 0.05) * (upper - lower) / r):
        N += 1
    N -= 1
    x1 = .0
    x2 = .0
    for i in range(N, 1, -1):
        L = upper - lower
        t1 = fibonacci_num(i - 2)
        t2 = fibonacci_num(i)
        x1 = lower + ((t1 / t2) * L)
        x2 = upper - (t1 / t2 * L)

        param2 = X - np.dot(x2, d)
        param2 = param2.tolist()

        param1 = X - np.dot(x1, d)
        param1 = param1.tolist()

        f1 = f.subs(list(zip(v, param1)))
        f2 = f.subs(list(zip(v, param2))) + (0.05 if i == 2 else 0.0)

        if f1 > f2:
            lower = x1
        else:
            upper = x2
    return x1


def Gradient(f, X=None):
    v = list(ordered(f.free_symbols))
    grd = lambda fn, vn: Matrix([fn]).jacobian(vn)
    if X:
        return list(grd(f, v).subs(list(zip(v, X))))
    else:
        return grd(f, v)


def conjugate_gradient(f, X, iterations, epslon, formula):
    v = list(ordered(f.free_symbols))
    xk = X
    # c2 = 0.1
    #print("x={}, f(x)={}".format(xk, f.subs(list(zip(v, xk)))))

    grad_f = Gradient(f)
    gk = np.array(list(grad_f.subs(list(zip(v, xk)))), dtype=np.float)
    dk = -gk

    for i in range(iterations):
        #alpha = golden_section_searcher(f, xk, -dk, 1, -5, 5, 0.0005)
        alpha = secant_search(grad_f, xk, dk)
        #print('alpha: ', alpha)
        #alpha = 0.02
        xk1 = xk + alpha * dk
        gk1 = np.array(list(grad_f.subs(list(zip(v, xk1)))), dtype=np.float)
        if formula == Hestenes_Stiefel:
            beta_k1 = np.dot(gk1, (gk1-gk)) / np.dot(dk, (gk1-gk))
        elif formula == Polak_Ribiere:
            beta_k1 = np.dot(gk1, (gk1-gk)) / np.dot(gk, gk)
        elif formula == Fletcher_Reeves:
            beta_k1 = np.dot(gk1, gk1) / np.dot(gk, gk)
            #print('beta_k: ', beta_k1)
        else:
            raise ValueError("Illegal value of the argument 'formula'.")
        dk1 = -gk1 + (beta_k1 * dk)
        if np.linalg.norm(xk1 - xk) < epslon * np.linalg.norm(xk1):
            xk = xk1
            break

        xk = xk1
        gk = gk1
        dk = dk1
        #if i % 1 == 0:
            # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, xk, f(xk))
            # print("  iter={}, x={}, f(x)={}".format(i, xk, f.subs(list(zip(v, xk)))))

    return xk

if __name__ == '__main__':
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    k, m, n = sp.symbols('k m n', integer=True)
    f, g, h = sp.symbols('f g h', cls=sp.Function)
    C = CoordSys3D('C', variable_names=('x1', 'x2', 'x3'))
    sp.init_printing(use_unicode=True)
    # f = 100*(x2 - x1**2)**2 + (1 - x1)**2
    f = (x1 - 1) ** 2 + (2 - x2 ** 2) ** 2 + 4  # * (x3 - 3)**4
    v = list(ordered(f.free_symbols))

    print(conjugate_gradient(f, [-1, 2], 1000, 0.00001, Hestenes_Stiefel))
    print(conjugate_gradient(f, [-1, 2], 1000, 0.00001, Polak_Ribiere))
    print(conjugate_gradient(f, [-1, 2], 1000, 0.00001, Fletcher_Reeves))

