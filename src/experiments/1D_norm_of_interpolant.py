import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import get_polynomial_as_tt, get_trig_function_as_tt
from laplace_mps.solver import evaluate_nodal_basis, get_rhs_matrix_as_tt, to_legendre_basis, _get_gram_matrix_legendre
from laplace_mps.tensormethods import TensorTrain

def build_mass_matrix(L):
    factors = [(np.eye(2)/2)[None, ..., None] for _ in range(L)]
    factors += [_get_gram_matrix_legendre()[None, ..., None] / 2]
    return TensorTrain(factors)

def get_L2_norm_in_legendre_basis(u):
    M = build_mass_matrix(L)
    return (u @ M @ u).squeeze().eval().flatten()

L_values = np.arange(3, 30)
error_L2 = []
refnorm_L2 = 14 / 15 + 12 / np.pi ** 4 - 4 / np.pi ** 2

for L in L_values:
    u = get_polynomial_as_tt([2,-2, 1], L) * get_trig_function_as_tt([0,1,0.25], L)
    u = to_legendre_basis(u)
    error_L2.append(get_L2_norm_in_legendre_basis(u) - refnorm_L2)

plt.close("all")
plt.semilogy(L_values, np.abs(error_L2), marker='o')
plt.grid(alpha=0.5)



