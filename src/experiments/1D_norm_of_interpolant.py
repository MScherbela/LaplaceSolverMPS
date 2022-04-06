import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import  evaluate_nodal_basis, get_example_u_1D
from laplace_mps.solver import get_laplace_matrix_as_tt, get_derivative_matrix_as_tt, get_L2_norm_1D

L_values = np.arange(3, 45)
error_L2 = np.ones_like(L_values, dtype=float)
error_H1 = np.ones_like(L_values, dtype=float)
error_H1_smart = np.ones_like(L_values, dtype=float)

refnorm_L2 = -3/(16*np.pi**2) + 3/(16*np.pi**4) + 1/15
refnorm_H1 = -1/(4*np.pi**2) + 5/12 + 4*np.pi**2/15

plt.close("all")
fig, axes = plt.subplots(1,2, dpi=100, figsize=(9,5))

for i, L in enumerate(L_values):
    A = get_laplace_matrix_as_tt(L)
    D = get_derivative_matrix_as_tt(L)
    u = get_example_u_1D(L)
    u = evaluate_nodal_basis(u, [1.0], basis='corner').squeeze()
    Du = D @ u
    error_L2[i] = get_L2_norm_1D(u) - refnorm_L2
    error_H1[i] = (u @ A @ u).squeeze().eval() - refnorm_H1
    error_H1_smart[i] = Du.norm_squared()*0.5**L - refnorm_H1

    if L == 10:
        axes[0].plot((np.arange(2**L) + 1)/2**L, u.eval(reshape='vector'))

axes[1].semilogy(L_values, np.abs(error_L2) / refnorm_L2, marker='o', label='L2')
axes[1].semilogy(L_values, np.abs(error_H1_smart) / refnorm_H1, marker='s', label='H1 orthogonalized')
axes[1].semilogy(L_values, np.abs(error_H1) / refnorm_H1, marker='^', label='H1 naive', ls='-')
axes[1].semilogy(L_values, 0.1 * 0.5 ** (2*L_values), label='~$2^{-2L}$', color='dimgray', zorder=-1)
for ax in axes:
    ax.grid(alpha=0.5)
axes[1].legend()

axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x)")
axes[0].set_title("Interpolation at L=10")
axes[1].set_xlabel("Refinement level L")
axes[1].set_ylabel("$\\frac{\\left| |u|^2 - |u|^2_{ref} \\right|}{|u|^2_{ref}}$")
axes[1].set_title("Error of norm")

fig.tight_layout()
fig.savefig("outputs/1D_norm_of_interpolant.pdf", bbox_inches='tight')



