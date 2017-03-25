"""Generates gradients using the CAM02-UCS colorspace."""

import io
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import theano
import theano.tensor as T
import ucs
from ucs.constants import floatX, Surrounds

from gradient_maker.optimizer import AdamOptimizer

# Function compiled in make_gradient() that should persist across instances of Gradient
opfunc = None


class BgColors:
    """Specifies three different background colors that work well with CIECAM02."""
    DARK = floatX([0.2]*3)
    NEUTRAL = floatX([0.5]*3)
    LIGHT = floatX([0.8]*3)


class Gradient:
    """Generates gradients using the CAM02-UCS colorspace."""
    def __init__(self, x, y, periodic=False, bg=BgColors.NEUTRAL, compile_only=False):
        if compile_only:
            self.x, self.y, self.bg = None, None, None
            self.make_gradient()
            return
        self.y = np.atleast_2d(y)
        if self.y.ndim != 2 or self.y.shape[-1] != 3:
            raise ValueError('y.ndim must be 2 and y.shape[-1] must be 3 (RGB).')
        self.x = np.linspace(0, 1, len(y)) if x is None else floatX(x)
        self.periodic = periodic
        self.bg = bg

    @staticmethod
    def compile():
        Gradient(None, None, compile_only=True)

    def make_gradient(self, steps=30, bg=BgColors.NEUTRAL, diff_weight=1e4, callback=None):
        global opfunc

        start = time.perf_counter()

        def _loss(y, ideal_jab, ideal_diff):
            jab = ucs.symbolic.srgb_to_ucs(y, 100, 20, ucs.srgb_to_xyz(bg)[1] * 100,
                                           **Surrounds.AVERAGE)
            diff = jab[1:, :] - jab[:-1, :]
            ucs_loss = T.sum(T.sqr(jab - ideal_jab))
            diff_loss = T.mean(T.sqr(diff - ideal_diff))
            return ucs_loss + diff_loss * diff_weight

        if opfunc is None:
            rgb, _ideal_jab, _ideal_diff = T.matrices('rgb', 'ideal_jab', 'ideal_diff')
            loss_sym = _loss(rgb, _ideal_jab, _ideal_diff)
            grad_sym = T.grad(loss_sym, rgb)

            # Ensure this function is compiled
            ucs.srgb_to_ucs([1, 1, 1])

            print('Building opfunc()...', file=sys.stderr)
            opfunc = theano.function([rgb, _ideal_jab, _ideal_diff], [loss_sym, grad_sym],
                                     allow_input_downcast=True, on_unused_input='ignore')
            print('Done building functions in {:.3g} seconds.'.format(time.perf_counter() - start),
                  file=sys.stderr)

        # If the method was called only to precompile Theano functions, return early
        if self.x is None:
            return

        jmh = ucs.jab_to_jmh(ucs.srgb_to_ucs(self.y, Y_b=ucs.srgb_to_xyz(self.bg)[1] * 100))
        jmh[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(jmh[:, 2])))
        if self.periodic:
            jmh[-1] = jmh[0]
            interp = CubicSpline(self.x, jmh, axis=0, bc_type='periodic')
        else:
            interp = PchipInterpolator(self.x, jmh, axis=0)
        ideal_jmh = np.zeros((steps, 3))
        x = np.linspace(self.x[0], self.x[-1], steps)
        for i, n in enumerate(x):
            ideal_jmh[i] = interp(n)
        ideal_jab = ucs.jmh_to_jab(ideal_jmh)
        ideal_diff = ideal_jab[1:, :] - ideal_jab[:-1, :]

        y = floatX(np.random.uniform(-1e-8, 1e-8, size=ideal_jab.shape)) + 0.5
        opt = AdamOptimizer(y, opfunc=lambda y: opfunc(y, ideal_jab, ideal_diff),
                            proj=lambda y: np.clip(y, 0, 1))

        for i in opt:
            if i % 100 == 0:
                loss_ = float(opfunc(y, ideal_jab, ideal_diff)[0])
                if callback is not None:
                    callback('Iteration {:d}, loss = {:.3f}'.format(i, loss_))

        done = time.perf_counter()
        s = ('Loss was {:.3f} after {:d} iterations; make_gradient() took {:.3f} seconds.').format(
            float(opfunc(y, ideal_jab, ideal_diff)[0]), i, done - start,
        )
        return x, y, s

    @staticmethod
    def print_stdout(x, y):
        """Prints the gradient to standard output. Requires a 24-bit color capable terminal."""
        for x_elem, elem in zip(x, y):
            rgb = np.uint8(np.round(elem*255))
            if ucs.srgb_to_xyz(elem)[1] < 1/2:
                print('\033[38;2;255;255;255m', end='')
            else:
                print('\033[38;2;0;0;0m', end='')
            print('\033[48;2;{};{};{}m'.format(*rgb), end='')
            print('%.04f' % x_elem, rgb, end='')
            print('\033[0m')

    @staticmethod
    def to_html(x, y):
        """Renders the gradient as HTML."""
        s = io.StringIO()
        s.write('<div class="gradient">\n')
        for x_elem, elem in zip(x, y):
            css_color = 'rgb({:.1f}%, {:.1f}%, {:.1f}%)'.format(*(elem * 100))
            if ucs.srgb_to_xyz(elem)[1] < 1/2:
                s.write('<div class="light-text" style="background-color: {};">'.format(css_color))
            else:
                s.write('<div class="dark-text" style="background-color: {};">'.format(css_color))
            s.write('{}</div>\n'.format(css_color))
        s.write('</div>\n')
        return s.getvalue()

    @staticmethod
    def to_csv(x, y):
        s = io.StringIO()
        s.write('x,r,g,b\n')
        for x_elem, (r, g, b) in zip(x, y):
            s.write(','.join(map(str, (x_elem, r, g, b))))
            s.write('\n')
        return s.getvalue()


def main():
    """A simple test case."""
    x = [0, 1/3, 1]
    y = floatX([[51, 51, 127], [51, 127, 51], [255, 102, 51]]) / 255
    # x = [0, 0.5, 1]
    # y = floatX([[20, 80, 120], [40, 60, 160], [255, 255, 200]]) / 255
    g = Gradient(x, y, bg=BgColors.LIGHT)
    x_out, y_out, _ = g.make_gradient(steps=30, bg=BgColors.LIGHT)
    g.print_stdout(x_out, y_out)

if __name__ == '__main__':
    main()
