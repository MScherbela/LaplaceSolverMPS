import numpy as np
from laplace_mps.solver import get_bpx_preconditioner_by_sum_2D, get_derivative_matrix_as_tt, get_overlap_matrix_as_tt, get_laplace_bpx_2D_by_sum, get_rhs_matrix_bpx_by_sum_2D
from laplace_mps.bpx2D import get_BPX_preconditioner_2D, get_BPX_Qp_2D, get_BPX_Q_2D, get_laplace_BPX_2D, get_rhs_matrix_BPX_2D

L = 3
def tensor_dot(A, B):
    C = A.copy().expand_dims([1,3]) * B.copy().expand_dims([0,2])
    new_shapes = [(U.shape[1]*U.shape[2], U.shape[3]*U.shape[4]) for U in C]
    return C.reshape_mode_indices(new_shapes)

def print_comparison(A, B, labelA, labelB):
    nA = A.norm_squared()
    nB = B.norm_squared()
    nRes = (A-B).norm_squared()
    print(f"Norm² {labelA: <20}: {nA:.2e}")
    print(f"Norm² {labelB: <20}: {nB:.2e}")
    print(f"Norm² {'residual': <20}: {nRes:.2e}")
    print(f"log2 {'ratio': <21}: {np.log2(nA/nB):8.3f}")
    print("-"*45)

C_2D = get_BPX_preconditioner_2D(L)
Q_2D = get_BPX_Q_2D(L)
Qp_2D = get_BPX_Qp_2D(L)
B_2D = get_laplace_BPX_2D(L)
R_2D = get_rhs_matrix_BPX_2D(L)

# Build references
B_2D_ref = get_laplace_bpx_2D_by_sum(L)
C_2D_ref = get_bpx_preconditioner_by_sum_2D(L)
R_2D_ref = get_rhs_matrix_bpx_by_sum_2D(L)
C_expanded = C_2D.copy()
C_expanded.tensors.append(np.ones([1,1,1,1]))
D = get_derivative_matrix_as_tt(L)
D.tensors.append(np.ones([1,1,1,1]))
M = get_overlap_matrix_as_tt(L)

Qp_2D_ref = tensor_dot(D, M) @ C_expanded
Q_2D_ref = tensor_dot(M, M) @ C_expanded

print_comparison(C_2D_ref, C_2D, "C by sum (old)", "C directly")
print_comparison(Qp_2D_ref, Qp_2D, "(M'xM) C by sum (old)", "(M'xM)C directly")
print_comparison(Q_2D_ref, Q_2D, "(MxM) C by sum (old)", "(MxM)C directly")
print_comparison(B_2D_ref, B_2D, "B by sum (old)", "B directly")
print_comparison(R_2D_ref, R_2D, "R by sum (old)", "R directly")


