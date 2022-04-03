import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import get_polynomial_as_tt, get_trig_function_as_tt, evaluate_nodal_basis
from laplace_mps.solver import _get_gram_matrix_legendre, \
    get_laplace_matrix_as_tt, get_derivative_matrix_as_tt, get_L2_norm_1D
from laplace_mps.tensormethods import TensorTrain

L_values = np.arange(3, 33)
error_L2 = np.ones_like(L_values, dtype=float)
error_H1 = np.ones_like(L_values, dtype=float)
error_H1_smart = np.ones_like(L_values, dtype=float)

refnorm_L2 = 14 / 15 + 12 / np.pi ** 4 - 4 / np.pi ** 2
refnorm_H1 = -1/3 - 1/np.pi**2 + 7/30*np.pi**2

plt.close("all")
fig, axes = plt.subplots(1,2, dpi=100, figsize=(13,8))

for i, L in enumerate(L_values):
    A = get_laplace_matrix_as_tt(L)
    D = get_derivative_matrix_as_tt(L)
    u = get_polynomial_as_tt([2,-2, 1], L) * get_trig_function_as_tt([0,1,0.25], L)
    u = evaluate_nodal_basis(u, [1.0], basis='corner').squeeze()
    Du = D @ u
    error_L2[i] = get_L2_norm_1D(u) - refnorm_L2
    error_H1[i] = (u @ A @ u).squeeze().eval() - refnorm_H1
    error_H1_smart[i] = Du.norm_squared()*0.5**L - refnorm_H1

    if L == 10:
        axes[0].plot((np.arange(2**L) + 1)/2**L, u.eval(reshape='vector'))


axes[1].semilogy(L_values, np.abs(error_L2) / refnorm_L2, marker='o', label='L2')
axes[1].semilogy(L_values, np.abs(error_H1_smart) / refnorm_H1, marker='s', label='H1 orthogonalized')
axes[1].semilogy(L_values, np.abs(error_H1) / refnorm_H1, marker='^', label='H1 directly', ls='--')
axes[1].semilogy(L_values, 0.1 * 0.5 ** (2*L_values), label='~$2^{-2L}$', color='dimgray', zorder=-1)
for ax in axes:
    ax.grid(alpha=0.5)
axes[1].legend()



