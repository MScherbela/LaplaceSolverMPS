from laplace_mps.solver import build_laplace_matrix_2D, get_bpx_preconditioner_by_sum_2D
import numpy as np
import matplotlib.pyplot as plt


eigenvalues_A_raw = []
eigenvalues_A_bpx = []
eigenvalues_bpx = []

rel_error = 1e-12
L_values = np.arange(2, 6)
for L in L_values:
    print(L)
    A = build_laplace_matrix_2D(L)
    C = get_bpx_preconditioner_by_sum_2D(L)
    AC = (A @ C).reapprox(rel_error=rel_error)
    A_bpx = (C @ AC).reapprox(rel_error=rel_error)

    A_eval = A.eval(reshape='matrix')
    A_bpx_eval = A_bpx.eval(reshape='matrix')
    C_eval = C.eval(reshape='matrix')
    eigenvalues_A_raw.append(np.linalg.eigvalsh(A_eval))
    eigenvalues_A_bpx.append(np.linalg.eigvalsh(A_bpx_eval))
    eigenvalues_bpx.append(np.linalg.eigvalsh(C_eval))

cond_nr_A_raw = [np.max(e)/np.min(e) for e in eigenvalues_A_raw]
cond_nr_A_bpx = [np.max(e)/np.min(e) for e in eigenvalues_A_bpx]
max_eigenvalue_bpx = [np.max(e) for e in eigenvalues_bpx]

plt.close("all")
plt.figure()
plt.semilogy(L_values, cond_nr_A_raw, label="Raw stiffness matrix")
plt.semilogy(L_values, cond_nr_A_bpx, label="Preconditioned stiffness matrix")
plt.semilogy(L_values, max_eigenvalue_bpx, label="Max. eigenvalue of preconditioner")
plt.xlabel("Nr of levels $L$")
plt.ylabel("Condition number")
plt.grid(alpha=0.5)
plt.legend()
