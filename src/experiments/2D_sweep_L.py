import os

print("Starting 2D_sweep_L.py")
from laplace_mps.solver import solve_PDE_2D_with_preconditioner, solve_PDE_2D, build_2D_mass_matrix, get_L2_norm_2D
import numpy as np
from laplace_mps.utils import get_example_u_2D, get_example_f_2D, _get_gram_matrix_tt, _get_identy_as_tt, kronecker_prod_2D, get_example_grad_u_2D
import datetime
import pandas as pd
import sys
import getpass

print(sys.argv)
if len(sys.argv) == 2:
    L_values = np.arange(int(sys.argv[1]), int(sys.argv[1])+1)
else:
    L_values = np.arange(2, 11)

error_L2 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_H1 = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_L2_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]
error_of_H1_norm = [np.ones(len(L_values))*np.nan, np.ones(len(L_values))*np.nan]

refnorm_L2 = (1/15 + 3 / (16*np.pi**4) - 3 / (16 * np.pi**2)) * (67 / 210)
refnorm_H1 = (2144*np.pi**6 + 5926*np.pi**4 + 7245 - 9255*np.pi**2)/(25200*np.pi**4)

rel_error = 1e-12
nswap = 150
max_rank = 150
kickrank = 2
print("eps=",rel_error)
print("nswap=",nswap)
print("max_rank=",max_rank)
print("kickrank=",kickrank)

data = []
for ind_solver, (is_bpx, solver) in enumerate(zip([True, False], [solve_PDE_2D_with_preconditioner, solve_PDE_2D])):
    for ind_L, L in enumerate(L_values):
        if not is_bpx and L > 14:
            break
        print(f"L = {L}")
        u_ref = get_example_u_2D(L, basis='nodal').reshape_mode_indices([4])
        f = get_example_f_2D(L).reapprox(rel_error=rel_error)
        DuDx_ref, DuDy_ref = get_example_grad_u_2D(L)
        u_solved, DuDx, DuDy = solver(f, max_rank=max_rank, eps=rel_error, nswp=nswap,  kickrank=kickrank)
        print(f"L = {L}: Calculating accuracy")

        # 2D gram matrix
        G = _get_gram_matrix_tt(L)
        G.tensors[-1] = np.sqrt(G.tensors[-1] / 2)
        I = _get_identy_as_tt(L, True)
        Gx = kronecker_prod_2D(G, I)
        Gy = kronecker_prod_2D(I, G)

        # |u-u0|L2
        delta_u = (u_solved - u_ref).reshape_mode_indices([4]).reapprox(rel_error=rel_error, ranks_new=max_rank)
        L2_residual = np.sqrt(get_L2_norm_2D(delta_u))
        print(f"L2 delta_u: {L2_residual:.4e}")

        # |u-u0|H1
        delta_DuDx = (DuDx - DuDx_ref).reapprox(rel_error=rel_error, ranks_new=max_rank)
        delta_DuDy = (DuDy - DuDy_ref).reapprox(rel_error=rel_error, ranks_new=max_rank)
        H1_residual = (0.25 ** L) * ((Gy @ delta_DuDx).norm_squared() + (Gx @ delta_DuDy).norm_squared())
        H1_residual = np.sqrt(H1_residual)
        print(f"H1 delta_u: {H1_residual:.4e}")

        # |u|H1 - |u0|H1
        H1_norm = (0.25 ** L) * ((Gy @ DuDx).norm_squared() + (Gx @ DuDy).norm_squared())
        error_H1_norm = H1_norm - refnorm_H1
        print(f"Error of H1 norm: {error_H1_norm:.4e}")

        # |u|L2 - |u0|L2
        error_L2_norm = get_L2_norm_2D(u_solved) - refnorm_L2
        print(f"Error of L2 norm: {error_L2_norm:.4e}")

        data.append(dict(L=L, bpx=is_bpx, L2_delta_u=L2_residual, H1_delta_u=H1_residual,
                         error_L2_norm=error_L2_norm, error_H1_norm=error_H1_norm))


if getpass.getuser() == 'scherbela':
    sys.exit(0)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
fname = f"/home/mscherbela/develop/LaplaceSolverMPS/outputs/2D_sweep_{timestamp}_nswp{nswap}_rank{max_rank}_eps{rel_error:.1e}_kickrank{kickrank}_.csv"

df = pd.DataFrame(data)
df.to_csv(fname, index=False)

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# fname = "/home/mscherbela/develop/LaplaceSolverMPS/outputs/vsc3_nswp150_rank600_eps1e-10_kickrank2.csv"
fname = "/home/mscherbela/develop/LaplaceSolverMPS/outputs/run4_eps1e-9_.csv"

df = pd.read_csv(fname)
df.sort_values(['bpx', 'L'], inplace=True)
eps = float(fname.split('eps')[-1].split('_')[0])

plt.close("all")
fig, axes = plt.subplots(1,2, dpi=100, figsize=(14,7))
for ind_solver, solver in enumerate(["With BPX precond", "without BPX"]):
    df_filt = df[df.bpx == (ind_solver==0)]
    if len(df_filt) == 0:
        continue
    color_L2 = ['red', 'C0'][ind_solver]
    color_H1 = ['salmon', 'lightblue'][ind_solver]
    ls = ['-', '--'][ind_solver]
    color = f'C{ind_solver}'
    axes[0].semilogy(df_filt.L, df_filt.L2_delta_u, marker='o', label=f"{solver}: L2", color=color_L2, ls=ls)
    if np.any(~np.isnan(df_filt.H1_delta_u)):
        axes[0].semilogy(df_filt.L, df_filt.H1_delta_u, marker='o', label=f"{solver}: H1", color=color_H1, ls=ls)

    axes[0].set_ylim([1e-10, None])
    axes[0].set_xlabel("L")
    axes[0].set_ylabel("norm of error")

    axes[1].semilogy(df_filt.L, np.abs(df_filt.error_L2_norm), marker='o', label=f"{solver}: L2", color=color_L2, ls=ls)
    if np.any(~np.isnan(df_filt.error_H1_norm)):
        axes[1].semilogy(df_filt.L, np.abs(df_filt.error_H1_norm), marker='o', label=f"{solver}: H1", color=color_H1, ls=ls)

axes[0].set_title("Norm of residual: $||u-u_{ref}||$")
axes[1].set_title("Error of norm: $||u||^2 - ref$")

axes[0].semilogy(df_filt.L, 0.6 * 0.5 ** (2 * df_filt.L), label='~$2^{-2L}$', color='lightgray', zorder=-1)
axes[1].semilogy(df_filt.L, 0.1 * 0.5 ** (2 * df_filt.L), label='~$2^{-2L}$', color='lightgray', zorder=-1)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.axhline(eps, color='k', ls='--', label='Solver accuracy')

plt.savefig(f"outputs/2D_sweep_L.pdf", bbox_inches='tight')

