from laplace_mps.solver import evaluate_nodal_basis, solve_PDE_1D_with_preconditioner
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import draw_vertical_grid, build_u_with_correct_boundary_conditions, get_f_from_u, evaluate_f_from_u, \
    eval_function, get_u_function_as_tt

L = 18
h = 0.5**L
solver_mse = 1e-10
plot_functions = True and (L <= 12)

# poly,trig = build_u_with_correct_boundary_conditions([0,1,-2], [(0.1,0.1,2.5), (0.1, 0.2, 34.5)])
poly,trig = build_u_with_correct_boundary_conditions([0, 1, -2, 3], [(1.0, 0.0, 2.0), (1.0, 1.0, 4)])
# poly,trig = build_u_with_correct_boundary_conditions([0,1,-2], [(1.0, 0.0, 0.0)])

f = get_f_from_u(poly, trig, L)
u_tt = get_u_function_as_tt(poly, trig, L)
u_tt_right = evaluate_nodal_basis(u_tt, [1.0]).squeeze()
r2_accuracy_solver = solver_mse**2 * (2**L)
u_solved, r2_precond = solve_PDE_1D_with_preconditioner(f, n_steps_max=500, max_rank=20, print_steps=True, r2_accuracy=r2_accuracy_solver)
residual = (u_tt_right - u_solved).reapprox(rel_error=1e-12)
L2_residual = (residual @ residual).squeeze().eval()
mean_squared_error = np.sqrt(L2_residual * h)
print(f"MSE: {mean_squared_error:.2e}")


if plot_functions:
    s_values = np.array([1.0])
    x_right = np.arange(1, (2 ** L) + 1) * h
    x_values = (x_right[:, None] + (s_values - 1) / 2 * h).flatten()

    u_dense_eval = eval_function(poly, trig, x_values)
    u_tt_eval = evaluate_nodal_basis(u_tt, s_values).eval(reshape='vector')
    u_solved_eval = u_solved.eval(reshape='vector')

    f_dense_eval = evaluate_f_from_u(poly, trig, x_values)
    f_eval = evaluate_nodal_basis(f, s_values).eval(reshape='vector')

    plt.close("all")
    fig, (ax_u, ax_f, ax_du) = plt.subplots(3,1, figsize=(14,7), sharex=True)
    ax_u.plot(x_right, u_solved_eval, label="Solution of PDE")
    ax_u.plot(x_values, u_dense_eval, label="Dense evaluation (orig. function)", ls='--')
    ax_u.plot(x_values, u_tt_eval, label="Orig. function converted to TT", ls='--')
    ax_u.set_ylabel("u")

    ax_du.plot(x_values, u_tt_eval - u_dense_eval, label="Residual of original TT")
    ax_du.plot(x_right, u_solved_eval - u_dense_eval, label="Residual of PDE solution")
    ax_du.axhline(mean_squared_error, label='MSE', color='gray')
    ax_du.set_ylabel("$\Delta u$")

    ax_f.plot(x_values, f_eval, label="f (TT)")
    ax_f.plot(x_values, f_dense_eval, label="f (directly)", ls='--')
    ax_f.set_ylabel("f")

    for ax in [ax_u, ax_f, ax_du]:
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        draw_vertical_grid(L, ax)
