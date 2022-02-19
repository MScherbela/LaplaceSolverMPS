import distutils.errors
from laplace_mps.solver import get_polynomial_as_tt, evaluate_nodal_basis, get_trig_function_as_tt, solve_PDE_1D_with_preconditioner
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import eval_poly, draw_vertical_grid, TrigFunction
from typing import List
import laplace_mps.tensormethods as tm

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
    return u_tt#.reapprox(rel_error=1e-16)


L = 5
h = 2**(-L)

# poly,trig = build_u_with_correct_boundary_conditions([0,1,-2], [(0.1,0.1,2.5), (0.1, 0.2, 34.5)])
poly,trig = build_u_with_correct_boundary_conditions([0,1,-2], [(1.0, 0.0, 2.0)])

# f = get_f_from_u(poly, trig, L)

u_tt = get_u_function_as_tt(poly, trig, L)
# u_solved = solve_PDE_1D_with_preconditioner(f, n_steps=20, max_rank=30)


s_values = np.array([-1.0])
x_right = np.arange(1, (2 ** L) + 1) * h
x_values = (x_right[:, None] + (s_values - 1) / 2 * h).flatten()

u_dense_eval = eval_function(poly, trig, x_right)
u_tt_eval = evaluate_nodal_basis(u_tt, s_values).eval(reshape='vector')
# u_solved_eval = u_solved.eval(reshape='vector')
# f_eval = evaluate_nodal_basis(f, s_values).eval(reshape='vector')

plt.close("all")
fig, (ax_u, ax_f, ax_du) = plt.subplots(3,1, figsize=(14,7), sharex=True)
# ax_u.plot(x_right, u_solved_eval, label="Solution of PDE")
ax_u.plot(x_right, u_dense_eval, label="Dense evaluation (orig. function)", ls='--')
ax_u.plot(x_right, u_tt_eval, label="Orig. function converted to TT", ls='--')
ax_u.set_ylabel("u")

ax_du.plot(x_right, u_tt_eval - u_dense_eval, label="Residual of original TT")
# ax_du.plot(x_right, u_solved_eval - u_dense_eval, label="Residual of PDE solution")
ax_du.set_ylabel("$\Delta u$")



# ax_f.plot(x_right, f_eval, label="f (TT)")
ax_f.set_ylabel("f")



for ax in [ax_u, ax_f, ax_du]:
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    draw_vertical_grid(5, ax)





#
#
#
# trig_coefs = [1,1,1]
#
# # poly_direct = eval_poly(x_values, poly_coeffs)
# # poly_tt = get_polynomial_as_tt(poly_coeffs, L)
#
# poly_direct = eval_trig(trig_coefs, x_values)
# poly_tt = get_trig_function_as_tt([1, 1, 1], L)
# poly_tt_eval = evaluate_nodal_basis(poly_tt, s_values).eval(reshape='vector')
# poly_tt_approx = poly_tt.copy().reapprox(rel_error=1e-16)
# poly_tt_approx_eval = evaluate_nodal_basis(poly_tt_approx, s_values).eval().flatten()
#
# print(f"Full TT ranks  : {poly_tt.ranks}")
# print(f"Approx TT ranks: {poly_tt_approx.ranks}")
#
#
# plt.close("all")
# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)
# ax1.plot(x_values, poly_direct, color='k', label='Reference')
# ax1.plot(x_values, poly_tt_eval, color='C0', label='Exact TT')
# ax1.plot(x_values, poly_tt_approx_eval, color='C1', label='Approximation')
# ax1.legend()
# ax1.set_ylabel("Function")
#
# ax2.axhline(0, color='k')
# ax2.plot(x_values, poly_tt_eval - poly_direct, color='C0', label='Exact TT')
# ax2.plot(x_values, poly_tt_approx_eval - poly_direct, color='C1', label='Approximation')
# ax2.legend()
# ax2.set_ylabel("Residual")
#
# for ax in [ax1, ax2]:
#     draw_vertical_grid(min(L, 5), ax)





