# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D
from sympy import ordered, Matrix, hessian
from math import *

phi = (1.0 + sqrt(5.0)) / 2.0


# searches for best learning_rate
# learning_rate is named as val here
# we want to optimize the f(X - L*d) function here where L is learning rate
def golden_section_searcher(f, X, d, prev_val, lower, upper, epsilon):
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

def fibonacci_num(n):
    if n <= 2:
        return 1
    else:
        return fibonacci_num(n - 1) + fibonacci_num(n - 2)


def fibonacci_search(f, X, d, lower, upper, epslon):
    v = list(ordered(f.free_symbols))
    N = 0
    while fibonacci_num(N) < ((1 + 2 * 0.05) * (upper - lower) / .23):
        N += 1
    N -= 1
    x1 = .0
    x2 = .0

    l = 'a'

    for i in range(N):
        L = upper - lower
        t1 = fibonacci_num(N+2-i)
        t2 = fibonacci_num(N+3-i)
        if i != N:
            roh = 1 - (t1 / t2)
        else:
            roh = 0.5 - epslon
        x1 = lower + roh * L
        x2 = lower + (1-roh) *  L

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
    #print("alpha: ", x1)
    return x1



# derivation idea is similar to limit derivation
# but h doesnt go 0 it is close to zero
# for high dimensions, we take delf
# which has all derivatives
def Gradient(f, X=None):
    v = list(ordered(f.free_symbols))
    grd = lambda fn, vn: Matrix([fn]).jacobian(vn)
    if X:
        return list(grd(f, v).subs(list(zip(v, X))))
    else:
        return grd(f, v)

def Hessian(f, X=None):
    v = list(ordered(f.free_symbols))
    if X:
        return list(hessian(f, v).subs(list(zip(v, X))))
    else:
        return hessian(f, v)

def difference(X, Y):
    total = 0

    for i in range(len(X)):
        total = total + abs(X[i] - Y[i])
    total = total / len(X)

    return total


def steepest_descent(f, X, epsilon):
    while True:
        d = Gradient(f, X)
        x_pre = X
        #lr = golden_section_searcher(f, X, d, 1, -10, 10, 0.0001)
        lr = fibonacci_search(f, X, d, -10, 10, 0.000001)
        X = X - np.dot(lr, d)
        X = X.tolist()

        if difference(x_pre, X) < epsilon:
            return x_pre

def newton(f, X, epsilon=1e-5, repeat=int(1e5)):
    pt = np.array(X)
    step_length = .001
    grad = Gradient(f, None)
    hess = Hessian(f, None)
    v = list(ordered(f.free_symbols))

    for i in range(repeat):
        g = np.array(list(grad.subs(list(zip(v, pt.tolist())))), dtype=np.float)
        h = np.array(list(hess.subs(list(zip(v, pt.tolist())))), dtype=np.float).reshape((len(X), len(X)))
        inverted_h = np.linalg.inv(h)
        direction = - np.dot(inverted_h, g)

        #if verbose: _verbose(i, pt, direction, inverted_hessian, step_length, function(pt))

        length_of_gradient = np.linalg.norm(g, 2)
        if step_length < epsilon or length_of_gradient < epsilon:
            break

        pt = pt + direction * step_length

    return pt.tolist()

# f(x) = (x-1)^2
# f(x1,x2) = x1^2 + x2^2 - 2x1x2
# f(x1,x2) = x1^2 + 2x2^2 - 5x1
# f(x1,x2,x3) = (x1-2)^2 + (2x2-3)^2 + x3^2

if __name__ == '__main__':
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    k, m, n = sp.symbols('k m n', integer=True)
    f, g, h = sp.symbols('f g h', cls=sp.Function)
    C = CoordSys3D('C', variable_names=('x1', 'x2', 'x3'))
    sp.init_printing(use_unicode=True)
    #f = (x1 - 1)**2 + (2 - x2**2)**2 + 4 #* (x3 - 3)**4
    f = 100 * ((x2 - x1 ** 2) ** 2) + (1 - x1) ** 2
    v = list(ordered(f.free_symbols))
    #print(np.array(list(Hessian(f).subs(list(zip(v, [5.0, 6.0, 0.0]))))).reshape((3, 3)))


    #print(steepest_descent(f, X=[-2, 3], epsilon=0.00001))
    print(newton(f, X=[-2, 3], epsilon=0.00001))
    # print('\n' + 'Final interval:' + str(golden_section_searcher([0], [1], equation([2]), 0, 2, .005)))

    # print(fibonacci_search(equation, 0, 2, 0.005))
    # x= sp.Symbol('x')



    # fn = sp.lambdify(v, gradient(f), "numpy")
    # print(eval(f, v))

    # print(gradient(f, np.array([3,2,1])))

    # print(gradient(f).subs([(v[0], 1), (v[1], 2), (v[2], 3)]))



