import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial
import matplotlib.pyplot as plt

def eval_poly(x, coeffs):
    y = np.zeros_like(x)
    for k, c in enumerate(coeffs):
        y += c * x**k
    return y

def draw_vertical_grid(L, ax=None):
    ax = ax or plt.gca()
    for i in range(2**L + 1):
        ax.axvline(i*2**(-L), color='k', alpha=0.1)


if __name__ == '__main__':
    poly_coefs = [1,2,3]
    p = numpy.polynomial.polynomial.Polynomial(poly_coefs)
    p_leg = p.convert(domain=[0,1], kind=numpy.polynomial.legendre.Legendre)
    print(p)
    print(p_leg)

