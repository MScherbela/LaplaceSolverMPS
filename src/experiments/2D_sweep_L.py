from laplace_mps.solver import evaluate_nodal_basis, solve_PDE_2D_with_preconditioner, solve_PDE_2D, build_2D_mass_matrix, get_L2_norm_2D
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import build_u_with_correct_boundary_conditions, get_example_u_2D, get_example_f_2D, get_example_f_1D


L_values = np.arange(2, 11)
error_L2 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_H1 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_L2_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_H1_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]

refnorm_L2 = (1/15 + 3 / (16*np.pi**4) - 3 / (16 * np.pi**2)) * (67 / 210)
refnorm_H1 = (2144*np.pi**6 + 5926*np.pi**4 + 7245 - 9255*np.pi**2)/(25200*np.pi**4)

for ind_solver, solver in enumerate([solve_PDE_2D_with_preconditioner, solve_PDE_2D]):
    for ind_L, L in enumerate(L_values):
        if (ind_solver == 1) and L > 10:
            break
        print(f"L = {L}")
        h = 0.5**(2*L)
        u_ref = get_example_u_2D(L, basis='nodal').reshape_mode_indices([4])
        f = get_example_f_2D(L).reapprox(rel_error=1e-15)
        u_solved = solver(f, eps=1e-10, nswp=20)
        print(f"L = {L}: Calculating accuracy")

        delta_u = (u_solved - u_ref).reshape_mode_indices([4]).reapprox(rel_error=1e-6, ranks_new=20)

        L2_residual = np.sqrt(get_L2_norm_2D(delta_u))
        error_of_L2_norm[ind_solver][ind_L] = get_L2_norm_2D(u_solved) - refnorm_L2

        print(f"L2: {L2_residual:.2e}")
        error_L2[ind_solver][ind_L] = L2_residual

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

    axes[0].set_ylim([1e-10, None])
    axes[0].set_xlabel("L")
    axes[0].set_ylabel("norm of error")

    axes[1].semilogy(L_values, np.abs(error_of_L2_norm[ind_solver]), marker='o', label=f"L2: {solver}", color=color_L2, ls=ls)
    axes[1].semilogy(L_values, np.abs(error_of_H1_norm[ind_solver]), marker='o', label=f"H1: {solver}", color=color_H1, ls=ls)

axes[0].set_title("Norm squared of residual: $||u-u_{ref}||$")
axes[1].set_title("Error of norm: $||u||^2 - ref$")

axes[0].semilogy(L_values, 0.6 * 0.5 ** (2 * L_values), label='~$2^{-2L}$', color='lightgray', zorder=-1)
axes[1].semilogy(L_values, 0.1 * 0.5 ** (2 * L_values), label='~$2^{-2L}$', color='lightgray', zorder=-1)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')

plt.savefig(f"outputs/2D_sweep_L.pdf", bbox_inches='tight')

