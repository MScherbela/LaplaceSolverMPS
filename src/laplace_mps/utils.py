from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

from laplace_mps import tensormethods as tm
from laplace_mps.solver import get_trig_function_as_tt, get_polynomial_as_tt, evaluate_nodal_basis

REL_ERROR = 1e-15

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


def get_example_u_1D(L, basis='corner'):
    u = get_polynomial_as_tt([1, -3, 2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    if basis == 'corner':
        s = np.array([-1, 1])
    elif basis == 'nodal':
        s = np.array([1])
    else:
        raise ValueError(f"Unknown basis: {basis}")
    return evaluate_nodal_basis(u, s).squeeze().reapprox(rel_error=REL_ERROR)

def get_example_u_deriv_1D(L, basis='corner'):
    u1 = get_polynomial_as_tt([-3, 4], L) * get_trig_function_as_tt([0, 1, 1.0], L) + \
        get_polynomial_as_tt([2*np.pi, -6*np.pi, 4*np.pi], L) * get_trig_function_as_tt([1, 0, 1.0], L)
    if basis == 'corner':
        s = np.array([-1, 1])
    elif basis == 'nodal':
        s = np.array([1])
    else:
        raise ValueError(f"Unknown basis: {basis}")
    return evaluate_nodal_basis(u1, s).squeeze().reapprox(rel_error=REL_ERROR)

def get_example_f_1D(L):
    f = get_polynomial_as_tt([4-4*np.pi**2, 12*np.pi**2, -8*np.pi**2], L) * get_trig_function_as_tt([0, 1, 1.0], L) + \
          get_polynomial_as_tt([-12*np.pi, 16*np.pi], L) * get_trig_function_as_tt([1, 0, 1.0], L)
    return -f.reapprox(rel_error=REL_ERROR)


def get_example_u_2D(L, basis='corner'):
    ux = get_polynomial_as_tt([0, -1, 5, -3], L)
    uy = get_polynomial_as_tt([1, -3, 2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    u = ux.expand_dims(1) * uy.expand_dims(0)
    if basis == 'corner':
        s = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]])
    elif basis == 'nodal':
        s = np.ones([2,1])
    else:
        raise ValueError(f"Unknown basis: {basis}")
    return evaluate_nodal_basis(u, s).squeeze()

def get_example_f_2D(L):
    ux = get_polynomial_as_tt([0, -1, 5, -3], L)
    uy = get_polynomial_as_tt([1, -3, 2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    ux2 = get_polynomial_as_tt([10, -18], L)
    uy2 = get_polynomial_as_tt([4-4*np.pi**2, 12*np.pi**2, -8*np.pi**2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    uy2 = uy2 + get_polynomial_as_tt([-12*np.pi, 16*np.pi], L) * get_trig_function_as_tt([1, 0, 1.0], L)

    f = ux.expand_dims(1) * uy2.expand_dims(0) + ux2.expand_dims(1) * uy.expand_dims(0)
    return -f.reapprox(rel_error=REL_ERROR)