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
L_values = np.arange(2, 12)
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
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9,5), dpi=100)
axes[0].semilogy(L_values, cond_nr_A_raw, label="$A$: Raw stiffness matrix", color='C0', lw=3)
axes[1].semilogy(L_values, cond_nr_A_bpx, label="$B$: Preconditioned stiffness matrix", color='C0', lw=3)
axes[0].semilogy(L_values, cond_nr_D, label="$D$: Derivative", color='C1', lw=3)
axes[1].semilogy(L_values, cond_nr_Qp, label="$Q'$: Derivative * BPX preconditioner", color='C1', lw=3)
axes[0].semilogy(L_values, cond_nr_M, label="$M$: Overlap", color='C2', lw=3)
axes[1].semilogy(L_values, cond_nr_Q, label="$Q$: Overlap * BPX preconditioner", color='C2', lw=3)
axes[1].semilogy(L_values, cond_nr_bpx, label="$C$: BPX preconditioner", color='C3', lw=3, ls='--')
axes[0].semilogy(L_values, 2 ** L_values, label="$2^L$", color='dimgray')
axes[1].semilogy(L_values, 2 ** L_values, label="$2^L$", color='dimgray')
axes[0].semilogy(L_values, 4 ** L_values, label="$2^{2L}$", color='lightgray')

for ax in axes:
    ax.set_xlabel("Nr of levels $L$")
    ax.grid(alpha=0.5)
    ax.legend(loc='upper left')
axes[0].set_ylabel("Condition number")
axes[0].set_title("Without preconditioner")
axes[1].set_title("With BPX preconditioner")
fig.tight_layout()
fig.savefig(f"outputs/1D_condition_number.pdf", bbox_inches='tight')


