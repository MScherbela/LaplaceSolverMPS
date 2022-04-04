from laplace_mps.solver import get_laplace_matrix_as_tt, get_bpx_preconditioner, get_derivative_matrix_as_tt
import numpy as np
import matplotlib.pyplot as plt

singular_values_A_raw = []
singular_values_A_bpx = []
singular_values_bpx = []
singular_values_DC = []

rel_error = 1e-12
L_values = np.arange(2, 10)
for L in L_values:
    print(L)
    A = get_laplace_matrix_as_tt(L)
    C = get_bpx_preconditioner(L)
    D = get_derivative_matrix_as_tt(L)
    A_bpx = (C @ A @ C).reapprox(rel_error=rel_error)

    A_eval = A.eval(reshape='matrix')
    A_bpx_eval = A_bpx.eval(reshape='matrix')
    C_eval = C.eval(reshape='matrix')
    DC_eval = (D @ C).reapprox(rel_error=rel_error).eval(reshape='matrix')
    singular_values_A_raw.append(np.linalg.svd(A_eval, compute_uv=False))
    singular_values_A_bpx.append(np.linalg.svd(A_bpx_eval, compute_uv=False))
    singular_values_bpx.append(np.linalg.svd(C_eval, compute_uv=False))
    singular_values_DC.append(np.linalg.svd(DC_eval, compute_uv=False))

cond_nr_A_raw = [np.max(e) / np.min(e) for e in singular_values_A_raw]
cond_nr_A_bpx = [np.max(e) / np.min(e) for e in singular_values_A_bpx]
cond_nr_bpx = [np.max(e) / np.min(e) for e in singular_values_bpx]
cond_nr_DC = [np.max(e) / np.min(e) for e in singular_values_DC]
# max_eigenvalue_bpx = [np.max(e) for e in eigenvalues_bpx]

plt.close("all")
plt.figure()
plt.semilogy(L_values, cond_nr_A_raw, label="Raw stiffness matrix")
plt.semilogy(L_values, cond_nr_A_bpx, label="Preconditioned stiffness matrix")
plt.semilogy(L_values, cond_nr_bpx, label="BPX preconditioner")
plt.semilogy(L_values, cond_nr_DC, label="$Q'$: Derivative * BPX preconditioner")
plt.semilogy(L_values, 2**L_values, label="$2^L$", color='dimgray')
plt.semilogy(L_values, 4**L_values, label="$2^{2L}$", color='lightgray')

plt.xlabel("Nr of levels $L$")
plt.ylabel("Condition number")
plt.grid(alpha=0.5)
plt.legend()

plt.savefig(f"outputs/1D_condition_number.pdf", bbox_inches='tight')


