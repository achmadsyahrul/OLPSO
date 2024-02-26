import numpy as np

def rosenbrock(x):
    """Rosenbrock function"""
    # return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    x1, x2, x3 = x[0], x[1], x[2]  # Menerima array 1 dimensi
    return np.sum(100.0*(x2-x1**2.0)**2.0 + (1-x1)**2.0 + 100.0*(x3-x2**2.0)**2.0 + (1-x2)**2.0)

def demo_func(x):
    x1, x2 = x
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.e