from laplace_mps.solver import get_polynomial_as_tt, evaluate_nodal_basis
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import eval_poly, draw_vertical_grid

poly_coeffs = [1,2,-3,-4,7]
L = 8
h = 2**-L
x_left = np.arange(2**L) * h


s_values = np.linspace(-1,1,5)
x_values = (x_left[:, None] + (s_values+1)/2 * h).flatten()

poly_direct = eval_poly(x_values, poly_coeffs)
poly_tt = get_polynomial_as_tt(poly_coeffs, L)
poly_tt_eval = evaluate_nodal_basis(poly_tt, s_values).eval().flatten()
poly_tt_approx = poly_tt.copy().reapprox(rel_error=1e-16)
poly_tt_approx_eval = evaluate_nodal_basis(poly_tt_approx, s_values).eval().flatten()

print(f"Full TT ranks  : {poly_tt.ranks}")
print(f"Approx TT ranks: {poly_tt_approx.ranks}")


plt.close("all")
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)
ax1.plot(x_values, poly_direct, color='k', label='Reference')
ax1.plot(x_values, poly_tt_eval, color='C0', label='Exact TT')
ax1.plot(x_values, poly_tt_approx_eval, color='C1', label='Approximation')
ax1.legend()
ax1.set_ylabel("Function")

ax2.axhline(0, color='k')
ax2.plot(x_values, poly_tt_eval - poly_direct, color='C0', label='Exact TT')
ax2.plot(x_values, poly_tt_approx_eval - poly_direct, color='C1', label='Approximation')
ax2.legend()
ax2.set_ylabel("Residual")

for ax in [ax1, ax2]:
    draw_vertical_grid(min(L, 5), ax)





