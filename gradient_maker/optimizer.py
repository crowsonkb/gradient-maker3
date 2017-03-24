"""Function minimization."""

import numpy as np
from ucs.constants import EPS


class AdamOptimizer:
    """Implements the Adam algorithm for gradient descent. Modifications from standard Adam include
    projected gradient descent and adaptive restarting. The momentum parameters have been changed
    from the default b1=0.9, b2=0.999 as well."""
    def __init__(self, x, opfunc, proj=None, step_size=1e-3, b1=0.98, b2=0.998,
                 factr=1e-6, maxiter=10000):
        self.x = x
        self.opfunc = opfunc
        self.proj = proj
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.factr = factr
        self.maxiter = maxiter
        self.g1 = np.zeros_like(x)
        self.g2 = np.zeros_like(x)
        self.i = 0
        self.last_loss = np.inf
        self.last_grad = np.zeros_like(x)

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.maxiter:
            raise StopIteration('maxiter reached.')

        loss, grad = self.opfunc(self.x)
        if loss > self.last_loss or np.sum(self.last_grad * grad) < 0:
            self.g1[:] = 0
        self.g1[:] = self.b1*self.g1 + (1-self.b1)*grad
        self.g2[:] = self.b2*self.g2 + (1-self.b2)*grad**2
        old_x = self.x.copy()
        ss = self.step_size * np.sqrt(1-self.b2**self.i) / (1-self.b1**self.i)
        self.x -= ss * self.g1 / (np.sqrt(self.g2) + EPS)
        if self.proj is not None:
            self.x[:] = self.proj(self.x)
        if np.mean(abs(old_x - self.x)) < self.factr:
            raise StopIteration('params change below tolerance.')

        self.last_loss, self.last_grad[:] = loss, grad
        return self.i
