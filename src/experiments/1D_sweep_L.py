from laplace_mps.solver import evaluate_nodal_basis, solve_PDE_1D_with_preconditioner, solve_PDE_1D, build_mass_matrix_in_nodal_basis, get_L2_norm_1D
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import build_u_with_correct_boundary_conditions, get_example_u_1D, get_example_u_deriv_1D, get_example_f_1D


L_values = np.arange(3, 30)
error_L2 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_H1 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_L2_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_H1_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]

refnorm_L2 = -3/(16*np.pi**2) + 3/(16*np.pi**4) + 1/15
refnorm_H1 = -1/(4*np.pi**2) + 5/12 + 4*np.pi**2/15

for ind_solver, solver in enumerate([solve_PDE_1D_with_preconditioner, solve_PDE_1D]):
    for ind_L, L in enumerate(L_values):
        if (ind_solver == 1) and L > 20:
            break
        print(f"L = {L}")
        h = 0.5**(L)
        u_ref = get_example_u_1D(L, basis='nodal')
        u_deriv_ref = get_example_u_deriv_1D(L, basis='nodal')
        f = get_example_f_1D(L).reapprox(rel_error=1e-15)
        # u_solved, u_deriv_solved, r2_precond = solver(f, n_steps_max=100, max_rank=max_rank, print_steps=True, rel_accuracy=1e-24)
        u_solved, u_deriv_solved = solver(f)


        delta_u = (u_solved - u_ref).reapprox(rel_error=1e-6)
        delta_u_deriv = (u_deriv_solved - u_deriv_ref).reapprox(rel_error=1e-6)
        mass = build_mass_matrix_in_nodal_basis(L)
        L2_residual = np.sqrt((delta_u @ mass @ delta_u).squeeze().eval())
        H1_residual = np.sqrt(delta_u_deriv.norm_squared()*h)
        # H1_residual = (delta_u_deriv @ mass @ delta_u_deriv).squeeze().eval()
        # L2_residual = (u_ref - u_solved).reapprox(rel_error=1e-12).norm_squared() * h
        # H1_residual = (u_deriv_ref - u_deriv_solved).reapprox(rel_error=1e-12).norm_squared() * h

        mass_matrix = build_mass_matrix_in_nodal_basis(L)
        error_of_L2_norm[ind_solver][ind_L] = get_L2_norm_1D(u_solved) - refnorm_L2
        error_of_H1_norm[ind_solver][ind_L] = u_deriv_solved.norm_squared()*h - refnorm_H1

        print(f"L2: {L2_residual:.2e}")
        print(f"H1: {H1_residual:.2e}")
        error_L2[ind_solver][ind_L] = L2_residual
        error_H1[ind_solver][ind_L] = H1_residual

#%%
plt.close("all")
fig, axes = plt.subplots(1,2, dpi=100, figsize=(14,7))
for ind_solver, solver in enumerate(["with BPX precond.", "no precond."]):
    color_L2 = ['red', 'salmon'][ind_solver]
    color_H1 = ['C0', 'lightblue'][ind_solver]
    ls = ['-', '--'][ind_solver]
    color = f'C{ind_solver}'
    axes[0].semilogy(L_values, error_L2[ind_solver], marker='o', label=f"L2: {solver}", color=color_L2, ls=ls)
    axes[0].semilogy(L_values, error_H1[ind_solver], marker='o', label=f"H1: {solver}", color=color_H1, ls=ls)
    # axes[0].semilogy(L_values, 0.5 ** (4*L_values), label='~$2^{-4L}$', color='dimgray', zorder=-1)
    # axes[0].semilogy(L_values, 0.5 ** (L_values), label='$2^{-L/2}$', color='lightgray', zorder=-1)

    axes[0].set_ylim([1e-16, None])
    axes[0].set_xlabel("L")
    axes[0].set_ylabel("norm of error")

    axes[1].semilogy(L_values, np.abs(error_of_L2_norm[ind_solver]), marker='o', label=f"L2: {solver}", color=color_L2, ls=ls)
    axes[1].semilogy(L_values, np.abs(error_of_H1_norm[ind_solver]), marker='o', label=f"H1: {solver}", color=color_H1, ls=ls)

axes[0].set_title("Norm of residual: $||u-u_{ref}||$")
axes[1].set_title("Error of norm: $||u||^2 - ref$")

axes[0].semilogy(L_values, 0.6 * 0.5 ** (2 * L_values), label='~$2^{-2L}$', color='lightgray', zorder=-1)
axes[1].semilogy(L_values, 0.3 * 0.5 ** (2 * L_values), label='~$2^{-2L}$', color='lightgray', zorder=-1)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')

plt.savefig(f"outputs/1D_sweep_L.pdf", bbox_inches='tight')

