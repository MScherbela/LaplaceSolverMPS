from laplace_mps.solver import solve_PDE_2D, solve_PDE_2D_with_preconditioner
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import draw_vertical_grid, eval_function, get_example_u_2D, get_example_f_2D, evaluate_nodal_basis


def imshow(ax, x):
    ax.imshow(x, cmap='bwr', clim=np.array([-1, 1]) * np.max(np.abs(x)), origin='lower', extent=[0, 1, 0, 1])

L = 7
h = 0.5**L
plot_functions = True and (L <= 7)

max_rank = 60
u_ref = get_example_u_2D(L, basis='nodal').flatten_mode_indices()
f = get_example_f_2D(L).reapprox(ranks_new=max_rank)
#
# u_solved = solve_PDE_2D_with_preconditioner(f)
u_solved = solve_PDE_2D_with_preconditioner(f, eps=1e-10, nswp=20)
residual = (u_ref - u_solved).reapprox(rel_error=1e-12)
L2_residual = (residual @ residual).squeeze().eval()
mean_squared_error = np.sqrt(L2_residual * h**2)
print(f"MSE: {mean_squared_error:.2e}")


if plot_functions:
    u_ref_eval = u_ref.reshape_mode_indices([2,2]).evalm()
    u_sol_eval = u_solved.reshape_mode_indices([2,2]).evalm()
    f_eval = f.evalm()[1::2, 1::2]

    plt.close("all")
    fig, axes = plt.subplots(2,2, figsize=(14,8), dpi=100)
    imshow(axes[0][0], u_ref_eval.T)
    axes[0][0].set_title("Reference solution")

    imshow(axes[0][1], f_eval.T)
    axes[0][1].set_title("-$\\nabla^2 u$")

    imshow(axes[1][0], u_sol_eval.T)
    axes[1][0].set_title("PDE solution")

    imshow(axes[1][1], (u_sol_eval-u_ref_eval).T)
    axes[1][1].set_title("Residual: ")
