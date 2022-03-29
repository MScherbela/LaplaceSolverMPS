import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import build_laplace_matrix_2D, get_bpx_preconditioner_by_sum_2D, get_bpx_Qp_term, get_bpx_Qt_term, get_gram_matrix_as_tt
import laplace_mps.tensormethods as tm

rel_error = 1e-12

L = 3
A = build_laplace_matrix_2D(L).reapprox(rel_error=rel_error)
C = get_bpx_preconditioner_by_sum_2D(L).reapprox(rel_error=rel_error)
B_naive = (C @ A @ C).reapprox(rel_error=rel_error)
print(B_naive.shapes)

Qp_matrices = [get_bpx_Qp_term(L, l) for l in range(0, L+1)]
Qt_matrices = [get_bpx_Qt_term(L, l) for l in range(0, L+1)]
G = get_gram_matrix_as_tt(L)

B = tm.zeros([[2,2,2,2] for _ in range(L)])
for l in range(L+1):
    for m in range(L+1):
        Qp_term = Qp_matrices[l].copy().transpose() @ Qp_matrices[m]
        Q_term = Qt_matrices[l] @ G @ Qt_matrices[m].copy().transpose()





