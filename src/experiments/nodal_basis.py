import laplace_mps.tensormethods as tm
import laplace_mps.solver as solver
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt

import laplace_mps.utils
from laplace_mps.utils import draw_vertical_grid


L = 5
h = 2**-L

poly_f = Polynomial([2,-3,6,-6])
poly_u = -poly_f.integ(1)
poly_u.coef[0] = -np.sum(poly_u.coef[1:]) # enforce right boundary condition
poly_u = poly_u.integ(1)
poly_u.coef[0] = 0 # enforce left boundary condition

u = laplace_mps.utils.get_polynomial_as_tt(poly_u.coef, L)
u_hat = laplace_mps.utils.evaluate_nodal_basis(u, [1.0]).squeeze()
u_nodal = solver.hat_to_nodal_basis(u_hat)

x_right = np.arange(1, 2**L+1) * h
s_values = np.linspace(-1, 1, 5)
x_values = (x_right[:,None] + (s_values-1)* h / 2).flatten()

plt.close("all")
for func,label,ls in zip([u, u_nodal], ['u orig', 'u nodal'], ['-','--']):
    func_eval = laplace_mps.utils.evaluate_nodal_basis(func, s_values).eval().flatten()
    plt.plot(x_values, func_eval, label=label,ls=ls)
plt.legend

