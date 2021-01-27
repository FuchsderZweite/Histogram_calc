import numpy as np

class Fit:

    def fit_sin(x, A, B, C, D):
        return A * np.sin(C * x + D) + B

    def polynom_fit(x, A, B, C, D, E, F):
        return A * x ** 5 + B * x ** 4 + C * x ** 3 + D * x ** 2 + E * x + F
