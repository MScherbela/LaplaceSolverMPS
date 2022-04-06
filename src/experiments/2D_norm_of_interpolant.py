import matplotlib.pyplot as plt
from laplace_mps.utils import get_example_u_2D, kronecker_prod_2D, _get_gram_matrix_tt, _get_identy_as_tt
from laplace_mps.solver import build_laplace_matrix_2D, get_L2_norm_2D, get_derivative_matrix_as_tt, get_overlap_matrix_as_tt
import numpy as np

#fx = (-3x^3 + 5x^2 -x)
#fy = (2y^2 - 3y + 1) * sin(2*pi*y)
refnorm_L2 = (1/15 + 3 / (16*np.pi**4) - 3 / (16 * np.pi**2)) * (67 / 210)
refnorm_H1 = (2144*np.pi**6 + 5926*np.pi**4 + 7245 - 9255*np.pi**2)/(25200*np.pi**4)

L_values = np.arange(2, 40)
error_L2 = np.zeros_like(L_values, dtype=float)
error_H1 = np.zeros_like(L_values, dtype=float)
error_H1_ort = np.zeros_like(L_values, dtype=float)

max_rank = 40
for i,L in enumerate(L_values):
    print(f"L = {L}")
    u = get_example_u_2D(L, basis='nodal')
    u = u.flatten_mode_indices().reapprox(ranks_new=max_rank)
    A = build_laplace_matrix_2D(L)
    Au = (A @ u).reapprox(ranks_new=max_rank)

    L2_norm = get_L2_norm_2D(u)
    H1_norm = (u @ Au).squeeze().eval()
    error_L2[i] = L2_norm - refnorm_L2
    error_H1[i] = H1_norm - refnorm_H1

    # 2D gram matrix
    G = _get_gram_matrix_tt(L)
    G.tensors[-1] = np.sqrt(G.tensors[-1] / 2)
    I = _get_identy_as_tt(L, True)
    Gx = kronecker_prod_2D(G, I)
    Gy = kronecker_prod_2D(I, G)

    u_expanded = u.copy()
    u_expanded.tensors.append(np.ones([1,1,1,1]))
    D = get_derivative_matrix_as_tt(L)
    D.tensors.append(np.ones([1,1,1,1]))
    M = get_overlap_matrix_as_tt(L)
    DuDx = (kronecker_prod_2D(D, M) @ u_expanded).flatten_mode_indices()
    DuDy = (kronecker_prod_2D(M, D) @ u_expanded).flatten_mode_indices()
    h1 = (0.25 ** L) * ((Gy @ DuDx).norm_squared() + (Gx @ DuDy).norm_squared())
    error_H1_ort[i] = (0.25 ** L) * ((Gy @ DuDx).norm_squared() + (Gx @ DuDy).norm_squared()) - refnorm_H1


#%%
L = 8
u = get_example_u_2D(L, basis='nodal')
u_eval = u.eval(reshape='matrix')
x_slice, y_slice = 0.5, 0.25

plt.close("all")
fig = plt.figure(dpi=100, figsize=(8,8))
gs = fig.add_gridspec(2, 2)
axes = [None, None, None]
axes[0] = fig.add_subplot(gs[0, :])
axes[1] = fig.add_subplot(gs[1, 0])
axes[2] = fig.add_subplot(gs[1, 1])

# fig, axes = plt.subplots(1,3, dpi=100, figsize=(12,6), gridspec_kw={'width_ratios': [1.5,1,1]})
axes[0].imshow(u_eval.T, cmap='bwr', clim=np.array([-1, 1]) * np.max(u_eval), origin='lower', extent=[0, 1, 0, 1])
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].axvline(x_slice, color='C0', alpha=0.5)
axes[0].axhline(y_slice, color='C1', alpha=0.5)
axes[0].set_title("Interpolation at L=8")

axes[1].plot(np.linspace(0, 1, 2**L), u_eval[int(x_slice * 2 ** L), :], label="f(0.5, y)")
axes[1].plot(np.linspace(0, 1, 2**L), u_eval[:, int(y_slice * 2 ** L)], label="f(x, 0.25)")
axes[1].legend()
axes[1].grid(alpha=0.5)
axes[1].set_xlabel("x / y")
axes[1].set_title("Slice of TT interpolation at L=8")
axes[1].set_ylabel("u(x,y)")

axes[2].semilogy(L_values, np.abs(error_L2) / refnorm_L2, marker='o', label='L2')
axes[2].semilogy(L_values, np.abs(error_H1_ort) / refnorm_H1, marker='s', label='H1 orthogonalized')
axes[2].semilogy(L_values, np.abs(error_H1) / refnorm_H1, marker='^', label='H1 naive', ls='-')

axes[2].grid(alpha=0.5)
axes[2].legend()
axes[2].set_xlabel("L")
axes[2].set_ylabel("$\\frac{\\left| |u|^2 - |u|^2_{ref} \\right|}{|u|^2_{ref}}$")
axes[2].set_title("Relative error of norm")
plt.tight_layout()
plt.savefig("outputs/2D_norm_of_interpolant.pdf", bbox_inches='tight')
