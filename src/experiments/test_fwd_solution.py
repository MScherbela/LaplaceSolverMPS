import laplace_mps.tensormethods as tm
import laplace_mps.solver as solver
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import draw_vertical_grid
import time

L = 7
h = 2**-L


M = solver.get_overlap_matrix_as_tt(L)
rhs_matrix = solver.get_rhs_matrix_as_tt(L)
Mp = solver.get_derivative_matrix_as_tt(L)
A = solver.get_laplace_matrix_as_tt(L)



poly_f = Polynomial([2,-3,6,-6])
# poly_f = Polynomial([1,-2,3])
# poly_f = Polynomial([1,3])
poly_u = -poly_f.integ(1)
poly_u.coef[0] = -np.sum(poly_u.coef[1:]) # enforce right boundary condition
poly_u = poly_u.integ(1)
poly_u.coef[0] = 0 # enforce left boundary condition

x_right = np.arange(1, 2**L+1) * h
s_values = np.linspace(-1, 1, 5)
x_values = (x_right[:,None] + (s_values-1)* h / 2).flatten()

f = solver.get_polynomial_as_tt(poly_f.coef, L)
u = solver.get_polynomial_as_tt(poly_u.coef, L)
u_right = solver.evaluate_nodal_basis(u, [1.0]).squeeze()
lhs = A @ u_right
rhs = rhs_matrix @ f

t0 = time.time()
u_solved = solver.solve_PDE_1D(f, max_rank=20, print_steps=True, n_steps=500, lr=0.5)
u_solved = solver.hat_to_nodal_basis(u_solved)
t1 = time.time()
print(t1-t0)

# f1 = Mp @ f
# f2 = -A @ f
#
# f_eval = f.eval().flatten()
# f1_eval = f1.eval().flatten()
# f2_eval = f2.eval().flatten()
Mp_eval = Mp.eval().transpose(list(range(0,2*L,2)) + list(range(1,2*L,2))).reshape([2**L, 2**L])
A_eval = A.eval().transpose(list(range(0,2*L,2)) + list(range(1,2*L,2))).reshape([2**L, 2**L])
M_eval = M.eval().transpose([-1] + list(range(0,2*L,2)) + list(range(1,2*L,2))).reshape([2, 2**L, 2**L])

plt.close("all")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.imshow(Mp_eval)
ax2.imshow(A_eval)
ax3.imshow(M_eval[0])
ax4.imshow(M_eval[1])

#
# plt.figure()
# plt.plot(x, f_eval, label="f(x) via TT", color='C0')
# plt.plot(x, poly(x), label="f(x)", color='navy', alpha=0.5, ls='--')
#
# plt.plot(x, f1_eval, label="f'(x) via TT", color='C2')
# plt.plot(x, poly_1(x), label="f'(x)", color='green', alpha=0.5, ls='--')
# #
# plt.plot(x, poly_2(x), label="f''(x)", color='C1', alpha=0.5)
# plt.plot(x, f2_eval, label="f''(x) via TT", color='red', alpha=0.5, ls='--')
# plt.legend()
# draw_vertical_grid(L)

fig, (ax1, ax2) = plt.subplots(2,1)
lhs_eval = lhs.eval().flatten()
rhs_eval = rhs.eval().flatten()
ax1.plot(x_right, lhs_eval, label='LHS: Au')
ax1.plot(x_right, rhs_eval, label='RHS: Mf', ls='--')
ax1.legend()


for (func, label, color) in zip([f, u, u_solved], ['f', 'u (ground-truth)', 'u solved'], ['gray', 'k', 'C0']):
    func_eval = solver.evaluate_nodal_basis(func, s_values).eval().flatten()
    ax2.plot(x_values, func_eval, label=label, color=color)
ax2.legend()

for ax in [ax1,ax2]:
    draw_vertical_grid(min(L,5), ax)





