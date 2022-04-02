from laplace_mps.solver import build_laplace_matrix_2D, get_bpx_preconditioner_by_sum_2D, get_bpx_preconditioner
import numpy as np
import matplotlib.pyplot as plt

rel_error = 1e-12

def get_bpx_preconditioner_as_product_2D(L):
    C = get_bpx_preconditioner(L)
    C_2D = C.copy().expand_dims([1,3]) * C.copy().expand_dims([0,2])
    return C_2D.squeeze().reshape_mode_indices([4,4]).reapprox(rel_error=rel_error)


eigenvalues_A_raw = []
eigenvalues_A_bpx = []
eigenvalues_A_bpx_prod = []
eigenvalues_bpx = []

L_values = np.arange(2, 6)
for L in L_values:
    print(L)
    A = build_laplace_matrix_2D(L)
    C = get_bpx_preconditioner_by_sum_2D(L)
    C_prod = get_bpx_preconditioner_as_product_2D(L)
    AC = (A @ C).reapprox(rel_error=rel_error)
    A_bpx = (C @ AC).reapprox(rel_error=rel_error)

    AC_prod = (A @ C_prod).reapprox(rel_error=rel_error)
    A_bpx_prod = (C_prod @ AC).reapprox(rel_error=rel_error)

    A_eval = A.evalm()
    A_bpx_eval = A_bpx.evalm()
    A_bpx_prod_eval = A_bpx_prod.evalm()
    C_eval = C.evalm()
    eigenvalues_A_raw.append(np.linalg.svd(A_eval, compute_uv=False))
    eigenvalues_A_bpx.append(np.linalg.svd(A_bpx_eval, compute_uv=False))
    eigenvalues_A_bpx_prod.append(np.linalg.svd(A_bpx_prod_eval, compute_uv=False))
    eigenvalues_bpx.append(np.linalg.svd(C_eval, compute_uv=False))

cond_nr_A_raw = [np.max(e)/np.min(e) for e in eigenvalues_A_raw]
cond_nr_A_bpx = [np.max(e)/np.min(e) for e in eigenvalues_A_bpx]
cond_nr_A_bpx_prod = [np.max(e)/np.min(e) for e in eigenvalues_A_bpx_prod]
cond_nr_bpx = [np.max(e)/np.min(e) for e in eigenvalues_bpx]

plt.close("all")
plt.figure()
plt.semilogy(L_values, cond_nr_A_raw, label="Raw stiffness matrix")
plt.semilogy(L_values, cond_nr_A_bpx, label="Preconditioned stiffness matrix")
plt.semilogy(L_values, cond_nr_A_bpx_prod, label="Preconditioned stiffness matrix (by CxC)")
plt.semilogy(L_values, cond_nr_bpx, label="Preconditioner")
plt.xlabel("Nr of levels $L$")
plt.ylabel("Condition number")
plt.grid(alpha=0.5)
plt.legend()

plt.savefig(f"outputs/2D_condition_number.pdf", bbox_inches='tight')
