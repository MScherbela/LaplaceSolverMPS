from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial
import matplotlib.pyplot as plt

from laplace_mps import tensormethods as tm
from laplace_mps.solver import get_trig_function_as_tt, get_polynomial_as_tt


def eval_poly(x, coeffs):
    y = np.zeros_like(x)
    for k, c in enumerate(coeffs):
        y += c * x**k
    return y

def draw_vertical_grid(L, ax=None):
    L = min(L, 6)
    ax = ax or plt.gca()
    for i in range(2**L + 1):
        ax.axvline(i*2**(-L), color='k', alpha=0.1)


class TrigFunction:
    def __init__(self, a,b, omega):
        self.a  = a
        self.b = b
        self.omega = omega

    def eval(self, x):
        return self.a * np.cos(self.omega * x) + self.b * np.sin(self.omega * x)

    def derive(self):
        return TrigFunction(self.b * self.omega, -self.a * self.omega, self.omega)

    @property
    def coeffs(self):
        return self.a, self.b, self.omega / (2 * np.pi)


if __name__ == '__main__':
    poly_coefs = [1,2,3]
    p = numpy.polynomial.polynomial.Polynomial(poly_coefs)
    p_leg = p.convert(domain=[0,1], kind=numpy.polynomial.legendre.Legendre)
    print(p)
    print(p_leg)


def build_u_with_correct_boundary_conditions(poly_coeffs, trig_coeffs):
    trig_functions = [TrigFunction(c[0], c[1], c[2] * 2 * np.pi) for c in trig_coeffs]
    trig_right = np.sum([t.eval(1) for t in trig_functions])
    trig_prime_right = np.sum([t.derive().eval(1) for t in trig_functions])
    ratio = trig_prime_right / trig_right

    poly_coeffs[0] = 0
    coeff_sum = np.sum([(ratio + i) * poly_coeffs[i] for i in range(2, len(poly_coeffs))])
    poly_coeffs[1] = -coeff_sum / (1 + ratio)
    poly = np.polynomial.Polynomial(poly_coeffs)
    return poly, trig_functions


def get_f_from_u(poly: np.polynomial.Polynomial, trig: List[TrigFunction], L):
    trig0_tt = tm.zeros([(2,) for _ in range(L+1)])
    trig1_tt = tm.zeros([(2,) for _ in range(L+1)])
    trig2_tt = tm.zeros([(2,) for _ in range(L+1)])
    for t in trig:
        trig0_tt = trig0_tt + get_trig_function_as_tt(t.coeffs, L)
        trig1_tt = trig1_tt + get_trig_function_as_tt(t.derive().coeffs, L)
        trig2_tt = trig2_tt + get_trig_function_as_tt(t.derive().derive().coeffs, L)

    poly0_tt = get_polynomial_as_tt(poly.coef, L)
    poly1_tt = get_polynomial_as_tt(poly.deriv(1).coef, L)
    poly2_tt = get_polynomial_as_tt(poly.deriv(2).coef, L)

    f = trig0_tt * poly2_tt + 2 * trig1_tt * poly1_tt + trig2_tt * poly0_tt
    return -f.reapprox(rel_error=1e-16)


def evaluate_f_from_u(poly: np.polynomial.Polynomial, trig: List[TrigFunction], x):
    trig0_eval = np.zeros_like(x)
    trig1_eval = np.zeros_like(x)
    trig2_eval = np.zeros_like(x)
    for t in trig:
        trig0_eval += t.eval(x)
        trig1_eval += t.derive().eval(x)
        trig2_eval += t.derive().derive().eval(x)

    poly0_eval = poly(x)
    poly1_eval = poly.deriv(1)(x)
    poly2_eval = poly.deriv(2)(x)
    f = poly0_eval * trig2_eval + 2 * poly1_eval * trig1_eval + poly2_eval * trig0_eval
    return -f


def eval_function(poly, trig_functions, x):
    trig = np.zeros_like(x)
    for t in trig_functions:
        trig += t.eval(x)
    return trig * poly(x)


def get_u_function_as_tt(poly, trig: List[TrigFunction], L):
    u_tt = get_trig_function_as_tt(trig[0].coeffs, L)
    for t in trig[1:]:
        u_tt = u_tt + get_trig_function_as_tt(t.coeffs, L)
    u_tt = u_tt * get_polynomial_as_tt(poly.coef, L)
    return u_tt.reapprox(rel_error=1e-16)