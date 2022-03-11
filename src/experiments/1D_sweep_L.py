from laplace_mps.solver import evaluate_nodal_basis, solve_PDE_1D_with_preconditioner
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import build_u_with_correct_boundary_conditions, get_f_from_u, get_u_function_as_tt

L_values = np.arange(6, 20)

solver_mse = 1e-12
r2_precond_values = []
mse_u_values = []
for L in L_values:
    h = 0.5**(L)
    poly,trig = build_u_with_correct_boundary_conditions([0, 1, -2, 3], [(1.0, 0.0, 2.0), (1.0, 1.0, 4)])
    f = get_f_from_u(poly, trig, L)

    u_tt = get_u_function_as_tt(poly, trig, L)
    u_tt_right = evaluate_nodal_basis(u_tt, [1.0]).squeeze()
    r2_accuracy_solver = solver_mse**2 * (2**L)
    u_solved, r2_precond = solve_PDE_1D_with_preconditioner(f, n_steps_max=500, max_rank=10, print_steps=True, r2_accuracy=r2_accuracy_solver)
    residual = (u_tt_right - u_solved).reapprox(rel_error=1e-12)
    L2_residual = (residual @ residual).squeeze().eval()
    mean_squared_error = np.sqrt(L2_residual * h)
    print(f"MSE: {mean_squared_error:.2e}")
    r2_precond_values.append(r2_precond_values)
    mse_u_values.append(mean_squared_error)

plt.close("all")
plt.semilogy(L_values, mse_u_values, marker='o')
plt.xlabel("L")
plt.ylabel("L2 MSE")
plt.grid(alpha=0.3)
plt.savefig("outputs/1D_sweep_L.pdf", bbox_inches='tight')

