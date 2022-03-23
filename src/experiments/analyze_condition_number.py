from laplace_mps.solver import get_laplace_matrix_as_tt, get_bpx_preconditioner
import numpy as np
import matplotlib.pyplot as plt


eigenvalues_A_raw = []
eigenvalues_A_bpx = []
eigenvalues_bpx = []

L_values = np.arange(2, 12)
for L in L_values:
    print(L)
    A = get_laplace_matrix_as_tt(L)
    C = get_bpx_preconditioner(L)
    A_bpx = (C @ A @ C).reapprox(rel_error=1e-12)

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

#
#
# fig, (ax_raw, ax_bpx, ax_eig) = plt.subplots(1,3)
# ax_raw.imshow(A_eval)
# ax_bpx.imshow(A_bpx_eval)
# ax_eig.semilogy(eig_raw, label="Raw A")
# ax_eig.semilogy(eig_bpx, label="BPX preconditioned")
# ax_eig.legend()
#
#
#


