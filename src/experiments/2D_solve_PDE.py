from laplace_mps.solver import evaluate_nodal_basis, solve_PDE_1D_with_preconditioner, solve_PDE_2D
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import draw_vertical_grid, eval_function, get_example_u_2D, get_example_f_2D

L = 5
h = 0.5**L
solver_mse = 1e-10
plot_functions = True and (L <= 8)

u_ref = get_example_u_2D(L, basis='nodal').flatten_mode_indices()
f = get_example_f_2D(L)

r2_accuracy_solver = solver_mse**2 * (2**(2*L))
u_solved, r2_precond = solve_PDE_2D(f, n_steps_max=400, max_rank=40, print_steps=True, r2_accuracy=r2_accuracy_solver)
residual = (u_ref - u_solved).reapprox(rel_error=1e-12)
L2_residual = (residual @ residual).squeeze().eval()
mean_squared_error = np.sqrt(L2_residual * h)
print(f"MSE: {mean_squared_error:.2e}")


if plot_functions:
    u_ref_eval = u_ref.reshape_mode_indices([2,2]).evalm()
    u_sol_eval = u_solved.reshape_mode_indices([2,2]).evalm()
    f_eval = f.evalm()[1::2, 1::2]

    plt.close("all")
    fig, axes = plt.subplots(2,2, figsize=(14,8), dpi=100)
    axes[0][0].imshow(u_ref_eval.T, cmap='bwr', clim=np.array([-1, 1]) * np.max(u_ref_eval), origin='lower', extent=[0, 1, 0, 1])
    axes[0][0].set_title("Reference solution")

    axes[0][1].imshow(f_eval.T, cmap='bwr', clim=np.array([-1, 1]) * np.max(f_eval), origin='lower', extent=[0, 1, 0, 1])
    axes[0][1].set_title("Reference solution")

    axes[1][0].imshow(u_sol_eval.T, cmap='bwr', clim=np.array([-1, 1]) * np.max(u_sol_eval), origin='lower', extent=[0, 1, 0, 1])
    axes[1][0].set_title("PDE solution")
