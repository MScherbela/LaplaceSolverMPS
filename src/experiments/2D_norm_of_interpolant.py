import matplotlib.pyplot as plt
from laplace_mps.utils import get_example_u_2D
from laplace_mps.solver import build_2D_mass_matrix, build_laplace_matrix_2D
import numpy as np

#fx = (-3x^3 + 5x^2 -x)
#fy = (2y^2 - 3y + 1) * sin(2*pi*y)
refnorm_L2 = (1/15 + 3 / (16*np.pi**4) - 3 / (16 * np.pi**2)) * (67 / 210)
refnorm_H1 = (2144*np.pi**6 + 5926*np.pi**4 + 7245 - 9255*np.pi**2)/(25200*np.pi**4)

L_values = np.arange(3, 10)
error_L2 = np.zeros_like(L_values, dtype=float)
error_H1 = np.zeros_like(L_values, dtype=float)

max_rank = 20
for i,L in enumerate(L_values):
    print(f"L = {L}")
    u = get_example_u_2D(L, basis='nodal')
    u = u.flatten_mode_indices().reapprox(ranks_new=max_rank)
    mass_matrix = build_2D_mass_matrix(L)
    A = build_laplace_matrix_2D(L)
    Mf = (mass_matrix @ u).reapprox(ranks_new=max_rank)
    Af = (A @ u).reapprox(ranks_new=max_rank)

    L2_norm = (u @ Mf).squeeze().eval()
    H1_norm = (u @ Af).squeeze().eval()
    error_L2[i] = L2_norm - refnorm_L2
    error_H1[i] = H1_norm - refnorm_H1


L = 8
u = get_example_u_2D(L, basis='nodal')
u_eval = u.eval(reshape='matrix')
x_slice, y_slice = 0.5, 0.25

plt.close("all")
fig, axes = plt.subplots(1,3, dpi=100, figsize=(14,6), gridspec_kw={'width_ratios': [2,1,1]})
axes[0].imshow(u_eval.T, cmap='bwr', clim=np.array([-1, 1]) * np.max(u_eval), origin='lower', extent=[0, 1, 0, 1])
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].axvline(x_slice, color='C0', alpha=0.5)
axes[0].axhline(y_slice, color='C1', alpha=0.5)

axes[1].plot(np.linspace(0, 1, 2**L), u_eval[int(x_slice * 2 ** L), :], label="f(0.5, y)")
axes[1].plot(np.linspace(0, 1, 2**L), u_eval[:, int(y_slice * 2 ** L)], label="f(x, 0.25)")
axes[1].legend()
axes[1].grid(alpha=0.5)
axes[1].set_xlabel("x / y")
axes[1].set_ylabel("Slice of TT interpolation")

axes[2].semilogy(L_values, np.abs(error_L2) / refnorm_L2, marker='o', label='L2')
# axes[2].semilogy(L_values, np.abs(error_H1_smart) / refnorm_H1, marker='s', label='H1 orthogonalized')
axes[2].semilogy(L_values, np.abs(error_H1) / refnorm_H1, marker='^', label='H1 directly', ls='-')
axes[2].grid(alpha=0.5)
axes[2].legend()
axes[2].set_xlabel("L")
axes[2].set_ylabel("Error of norm^2")
plt.tight_layout()


