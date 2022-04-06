from laplace_mps.solver import build_laplace_matrix_2D, get_bpx_preconditioner, get_derivative_matrix_as_tt, get_overlap_matrix_as_tt
from laplace_mps.bpx2D import get_laplace_BPX_2D, get_BPX_preconditioner_2D, get_BPX_Qp_2D, get_BPX_Q_2D
import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import kronecker_prod_2D

rel_error = 1e-12

def get_bpx_preconditioner_as_product_2D(L):
    C = get_bpx_preconditioner(L)
    C_2D = C.copy().expand_dims([1,3]) * C.copy().expand_dims([0,2])
    return C_2D.squeeze().reshape_mode_indices([4,4]).reapprox(rel_error=rel_error)


def get_cond_nr(A):
    s = np.linalg.svd(A, compute_uv=False)
    return np.max(s) / np.min(s)

cond_nr_A_raw = []
cond_nr_A_bpx = []
cond_nr_A_bpx_prod = []
cond_nr_bpx = []
cond_nr_M = []
cond_nr_D = []
cond_nr_Q = []
cond_nr_Qp = []

L_values = np.arange(2, 6)
for L in L_values:
    print(L)
    A = build_laplace_matrix_2D(L)
    C = get_BPX_preconditioner_2D(L)
    C_prod = get_bpx_preconditioner_as_product_2D(L)
    A_bpx = get_laplace_BPX_2D(L)
    AC_prod = (A @ C_prod).reapprox(rel_error=rel_error)
    A_bpx_prod = (C_prod @ AC_prod).reapprox(rel_error=rel_error)
    Q = get_BPX_Q_2D(L)
    Qp = get_BPX_Qp_2D(L)
    Qp.evalm()

    D = get_derivative_matrix_as_tt(L)
    D.tensors.append(np.ones([1, 1, 1, 1]))
    M = get_overlap_matrix_as_tt(L)
    Dx = kronecker_prod_2D(D, M)
    M2D = kronecker_prod_2D(M, M)

    cond_nr_A_raw.append(get_cond_nr(A.evalm()))
    cond_nr_A_bpx.append(get_cond_nr(A_bpx.evalm()))
    cond_nr_A_bpx_prod.append(get_cond_nr(A_bpx_prod.evalm()))
    cond_nr_bpx.append(get_cond_nr(C.evalm()))
    cond_nr_D.append(get_cond_nr(Dx.evalm()))
    cond_nr_M.append(get_cond_nr(M2D.evalm()))
    cond_nr_Q.append(get_cond_nr(Q.evalm()))
    cond_nr_Qp.append(get_cond_nr(Qp.evalm()))

plt.close("all")
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9,5), dpi=100)
axes[0].semilogy(L_values, cond_nr_A_raw, label="$A$: Raw stiffness matrix", color='C0', lw=3)
axes[1].semilogy(L_values, cond_nr_A_bpx, label="$B$: Preconditioned stiffness matrix", color='C0', lw=3)
axes[0].semilogy(L_values, cond_nr_D, label="$D \otimes M$: Derivative matrix", color='C1', lw=3)
axes[1].semilogy(L_values, cond_nr_Qp, label="$Q' = (D \otimes M) C$", color='C1', lw=3)

axes[0].semilogy(L_values, cond_nr_M, label="$M \otimes M$: Overlap matrix", color='C2', lw=3)
axes[1].semilogy(L_values, cond_nr_Q, label="$Q = (M \otimes M) C$", color='C2', lw=3)
axes[1].semilogy(L_values, cond_nr_bpx, label="$C$: Preconditioner", color='C3', lw=3)

axes[0].semilogy(L_values, 2**L_values, label="$2^L$", color='dimgray', zorder=-1)
axes[1].semilogy(L_values, 2**L_values, label="$2^L$", color='dimgray', zorder=-1)
axes[0].semilogy(L_values, 0.8 * 4**L_values, label="$2^{2L}$", color='lightgray', zorder=-1)

for ax in axes:
    ax.set_xlabel("Nr of levels $L$")
    ax.grid(alpha=0.5)
    ax.legend()
    ax.set_xticks(L_values)
axes[0].set_ylabel("Condition number")
axes[0].set_title("Without preconditioner")
axes[1].set_title("With BPX preconditioner")
fig.tight_layout()
plt.savefig(f"outputs/2D_condition_number.pdf", bbox_inches='tight')
