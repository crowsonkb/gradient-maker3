"""Implements the Adam algorithm for gradient descent."""

import numpy as np
from ucs.constants import EPS


class AdamOptimizer:
    """Implements the Adam algorithm for gradient descent."""
    def __init__(self, x, grad_fn, proj_fn=None, step_size=1e-4, b1=0.9, b2=0.999,
                 factr=1e-6, maxiter=10000):
        self.x = x
        self.grad_fn = grad_fn
        self.proj_fn = proj_fn
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.factr = factr
        self.maxiter = maxiter
        self.g1 = np.zeros_like(x)
        self.g2 = np.zeros_like(x)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.maxiter:
            raise StopIteration('maxiter reached.')

        grad = self.grad_fn(self.x)
        self.g1[:] = self.b1*self.g1 + (1-self.b1)*grad
        self.g2[:] = self.b2*self.g2 + (1-self.b2)*grad**2
        old_x = self.x.copy()
        ss = self.step_size * np.sqrt(1-self.b2**self.i) / (1-self.b1**self.i)
        self.x -= ss * self.g1 / (np.sqrt(self.g2) + EPS)
        if self.proj_fn is not None:
            self.x[:] = self.proj_fn(self.x)
        if np.mean(abs(old_x - self.x)) < self.factr:
            raise StopIteration('params change below tolerance.')

        return self.i
