# Goudi Dwaish
# COT4500-23Spring 0001
# Bonus Assignment
# Professor Juna Parra

import numpy as np

# Gauss-Seidel method
def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)
    for k in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x - x0) < tol:
            return k+1, x
        x0 = np.copy(x)
    return k+1, x

# Jacobi method
def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x) < tol:
            return k+1, x_new
        x = np.copy(x_new)
    return k+1, x

# Newton-Raphson method
def f(x):
    return x**3 - x**2 + 2

def df(x):
    return 3*x**2 - 2*x

def newton_raphson(f, df, x0, tol):
    k = 0
    while abs(f(x0)) > tol:
        x0 = x0 - f(x0) / df(x0)
        k += 1
    return k

# Hermite interpolation
def divided_diff(x, y, dy):
    n = len(x)
    F = np.zeros((2*n, 2*n))
    F[::2, 0] = x
    F[1::2, 0] = x
    F[::2, 1] = y
    F[1::2, 1] = dy
    for j in range(2, 2*n):
        for i in range(2*n-j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (F[i+j, 0] - F[i, 0])
    return F

# Modified Euler's method
def f(t, y):
    return y - t**3

def modified_euler(f, y0, t0, tf, h):
    t = t0
    y = y0
    while t < tf:
        y_pred = y + h * f(t, y)
        y = y + h/2 * (f(t, y) + f(t+h, y_pred))
        t += h
    return y

# Question 1
A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]], dtype=float)
b = np.array([1, 3, 0], dtype=float)
x0 = np.array([0, 0, 0], dtype=float)
tol = 1e-6
max_iter = 50
n_iter, x = gauss_seidel(A, b, x0, tol, max_iter)
