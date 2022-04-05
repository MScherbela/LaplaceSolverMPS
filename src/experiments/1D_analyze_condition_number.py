from laplace_mps.solver import get_laplace_matrix_as_tt, get_bpx_preconditioner, get_derivative_matrix_as_tt, get_bpx_Qp, get_bpx_Qt, get_overlap_matrix_as_tt
import numpy as np
import matplotlib.pyplot as plt

cond_nr_A_raw = []
cond_nr_A_bpx = []
cond_nr_bpx = []
cond_nr_Qp = []
cond_nr_Q = []
cond_nr_D = []
cond_nr_M = []

def get_cond_nr(A):
    s = np.linalg.svd(A, compute_uv=False)
    return np.max(s) / np.min(s)

rel_error = 1e-12
L_values = np.arange(2, 9)
for L in L_values:
    print(L)
    A = get_laplace_matrix_as_tt(L)
    C = get_bpx_preconditioner(L)
    M = get_overlap_matrix_as_tt(L)
    D = get_derivative_matrix_as_tt(L)
    A_bpx = (C @ A @ C).reapprox(rel_error=rel_error)
    Qp = get_bpx_Qp(L)
    Q = get_bpx_Qt(L).transpose()


    cond_nr_A_raw.append(get_cond_nr(A.evalm()))
    cond_nr_A_bpx.append(get_cond_nr(A_bpx.evalm()))
    cond_nr_bpx.append(get_cond_nr(C.evalm()))
    cond_nr_D.append(get_cond_nr(D.evalm()))
    cond_nr_M.append(get_cond_nr(M.evalm()))
    cond_nr_Qp.append(get_cond_nr(Qp.evalm()))
    cond_nr_Q.append(get_cond_nr(Q.evalm()))


plt.close("all")
plt.figure()
plt.semilogy(L_values, cond_nr_A_raw, label="Raw stiffness matrix", color='C0', ls='-')
plt.semilogy(L_values, cond_nr_A_bpx, label="Preconditioned stiffness matrix", color='C0', ls='--')
plt.semilogy(L_values, cond_nr_D, label="$D$: Derivative", color='C1', ls='-')
plt.semilogy(L_values, cond_nr_Qp, label="$Q'$: Derivative * BPX preconditioner", color='C1', ls='--')
plt.semilogy(L_values, cond_nr_M, label="$M$: Overlap", color='C2', ls='-')
plt.semilogy(L_values, cond_nr_Q, label="$Q$: Overlap * BPX preconditioner", color='C2', ls='--')
plt.semilogy(L_values, cond_nr_bpx, label="$C$: BPX preconditioner", color='k', ls='--')

plt.semilogy(L_values, 2**L_values, label="$2^L$", color='dimgray')
plt.semilogy(L_values, 4**L_values, label="$2^{2L}$", color='lightgray')

plt.xlabel("Nr of levels $L$")
plt.ylabel("Condition number")
plt.grid(alpha=0.5)
plt.legend()

plt.savefig(f"outputs/1D_condition_number.pdf", bbox_inches='tight')


