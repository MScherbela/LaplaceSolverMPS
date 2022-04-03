from laplace_mps.solver import solve_PDE_1D_with_preconditioner, solve_PDE_1D, get_laplace_matrix_as_tt
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import draw_vertical_grid, build_u_with_correct_boundary_conditions, get_f_from_u, evaluate_f_from_u, \
    eval_function, get_u_function_as_tt, get_example_u_1D, get_example_u_deriv_1D, get_example_f_1D, evaluate_nodal_basis

L = 20
h = 0.5**L
plot_functions = True and (L <= 14)

max_rank = 20

u_ref = get_example_u_1D(L, basis='nodal')
u_deriv_ref = get_example_u_deriv_1D(L, basis='nodal')
f = get_example_f_1D(L).reapprox(ranks_new=max_rank)

# u_solved, u_deriv_solved, r2_precond = solve_PDE_1D_with_preconditioner(f, n_steps_max=100, max_rank=max_rank, print_steps=True, rel_accuracy=1e-12)
u_solved, u_deriv_solved = solve_PDE_1D_with_preconditioner(f)
residual = (u_ref - u_solved).reapprox(rel_error=1e-5)
L2_residual = (residual @ residual).squeeze().eval()
mean_squared_error = np.sqrt(L2_residual * h)
print(f"MSE: {mean_squared_error:.2e}")


if plot_functions:
    s_values = np.array([1.0])
    x_right = np.arange(1, (2 ** L) + 1) * h
    x_values = (x_right[:, None] + (s_values - 1) / 2 * h).flatten()


    u_ref_eval = u_ref.evalv()
    u_deriv_ref_eval = u_deriv_ref.evalv()
    u_solved_eval = u_solved.evalv()
    u_deriv_solved_eval = u_deriv_solved.evalv()
    f_eval = evaluate_nodal_basis(f, s_values).eval(reshape='vector')
    A = get_laplace_matrix_as_tt(L)
    f_solved_eval = -(A @ u_solved).reapprox(ranks_new=max_rank).evalv()

    plt.close("all")
    fig, axes = plt.subplots(3,1, figsize=(14,7), sharex=True, dpi=100)
    axes[0].plot(x_right, u_solved_eval, label="Solution of PDE")
    axes[0].plot(x_values, u_ref_eval, label="Reference solution", ls='--')
    axes[0].set_ylabel("u")

    axes[1].plot(x_right, u_deriv_solved_eval, label="Solution of PDE: $u'$", ls='-')
    axes[1].plot(x_right, u_deriv_ref_eval, label="Reference $u'$", ls='--')
    axes[1].set_ylabel("$u'$")

    axes[2].plot(x_values, f_eval, label="f input")
    # axes[2].plot(x_values, f_solved_eval, label="$-Au$", ls='--')
    axes[2].set_ylabel("f")

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        draw_vertical_grid(L, ax)
