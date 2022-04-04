from laplace_mps.solver import solve_PDE_2D_with_preconditioner, solve_PDE_2D, build_2D_mass_matrix, get_L2_norm_2D
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import build_u_with_correct_boundary_conditions, get_example_u_2D, get_example_f_2D, get_example_f_1D, \
    evaluate_nodal_basis, _get_gram_matrix_tt, _get_identy_as_tt, kronecker_prod_2D

L_values = np.arange(2, 12)
error_L2 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_H1 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_L2_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_H1_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]

refnorm_L2 = (1/15 + 3 / (16*np.pi**4) - 3 / (16 * np.pi**2)) * (67 / 210)
refnorm_H1 = (2144*np.pi**6 + 5926*np.pi**4 + 7245 - 9255*np.pi**2)/(25200*np.pi**4)

rel_error = 1e-10
for ind_solver, solver in enumerate([solve_PDE_2D_with_preconditioner, solve_PDE_2D][:1]):
    for ind_L, L in enumerate(L_values):
        if (ind_solver == 1) and L > 10:
            break
        print(f"L = {L}")
        u_ref = get_example_u_2D(L, basis='nodal').reshape_mode_indices([4])
        f = get_example_f_2D(L).reapprox(rel_error=rel_error)
        u_solved, DuDx, DuDy = solver(f, eps=rel_error, nswp=50, max_rank=200)
        print(f"L = {L}: Calculating accuracy")

        # |u-u0|L2
        delta_u = (u_solved - u_ref).reshape_mode_indices([4]).reapprox(rel_error=rel_error, ranks_new=100)
        L2_residual = np.sqrt(get_L2_norm_2D(delta_u))
        error_L2[ind_solver][ind_L] = L2_residual
        print(f"L2 delta_u: {L2_residual:.4e}")

        # |u|H1 - |u0|H1
        G = _get_gram_matrix_tt(L)
        G.tensors[-1] = np.sqrt(G.tensors[-1] / 2)
        I = _get_identy_as_tt(L, True)
        Gx = kronecker_prod_2D(G, I)
        Gy = kronecker_prod_2D(I, G)
        H1_norm = (0.25 ** L) * ((Gy @ DuDx).norm_squared() + (Gx @ DuDy).norm_squared())
        error_of_H1_norm[ind_solver][ind_L] = H1_norm - refnorm_H1
        print(f"Error of H1 norm: {error_of_H1_norm[ind_solver][ind_L]:.4e}")

        # |u|L2 - |u0|L2
        error_of_L2_norm[ind_solver][ind_L] = get_L2_norm_2D(u_solved) - refnorm_L2
        print(f"Error of L2 norm: {error_of_L2_norm[ind_solver][ind_L]:.4e}")

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

