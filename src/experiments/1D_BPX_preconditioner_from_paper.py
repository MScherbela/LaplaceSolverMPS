import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner, _get_bpx_factors, _get_refinement_tensor, get_bpx_preconditioner_by_sum_2D, get_derivative_matrix_as_tt, get_overlap_matrix_as_tt, get_bpx_Qp, get_bpx_Qt, get_bpx_Qt_term, get_bpx_Qp_term
import laplace_mps.tensormethods as tm

def mode_product(A,B):
    shape_out = (A.shape[0]*B.shape[0], A.shape[1], B.shape[2], A.shape[-1]*B.shape[-1])
    C = np.einsum("pabq,PbcQ->pPacqQ", A, B)
    return C.reshape(shape_out)

def transpose_core(A):
    return A.transpose([0,2,1,3])

def tensor_product(A,B):
    assert len(A.shape) == len(B.shape)
    A = A[:, None, :, None, :, None, :, None]
    B = B[None, :, None, :, None, :, None, :]
    C = A*B
    new_shape = [C.shape[i] * C.shape[i+1] for i in range(0, C.ndim, 2)]
    return C.reshape(new_shape)

def strong_kronecker(A,B):
    shape_out = [A.shape[0], A.shape[1]*B.shape[1], A.shape[2]*B.shape[2], B.shape[3]]
    C = np.einsum("pabq,qABr->paAbBr", A, B).reshape(shape_out)
    return C.reshape(shape_out)

def _get_U_hat():
    J = np.array([[0, 1], [0, 0]], dtype=float)
    U_hat = np.zeros([2, 2, 2, 2])
    U_hat[0, :, :, 0] = np.eye(2)
    U_hat[0, :, :, 1] = J.T
    U_hat[1, :, :, 1] = J
    return U_hat

def _get_X_hat():
    X_hat = np.zeros([2,2,2], float)
    X_hat[0,:,0] = [1,2]
    X_hat[0,:,1] = [0,1]
    X_hat[1,:,0] = [1,0]
    X_hat[1,:,1] = [2,1]
    return X_hat[:, :, None, :] / 2

def _get_Y0_hat():
    Y0_hat = np.zeros([2,2,2])
    Y0_hat[0,:,0] = [2,2]
    Y0_hat[1,:,0] = [-1,1]
    Y0_hat[1,:,1] = [1,1]
    return Y0_hat[:, :, None, :] / 2

def _get_Y1_hat():
    return np.ones([1, 2, 1, 1]) / 2

def _get_N0_hat():
    return np.array([1,0,0,1]).reshape([2,2,1,1]) / 2

def _get_N1_hat():
    return np.ones([1,1,1,1])

def _get_P_hat():
    return np.array([1,0]).reshape([2,1,1,1])

def _get_W0_hat():
    return np.array([[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]).reshape([4,1,1,4])

def _get_W1_hat():
    return np.array([[1,0],[0,1],[-1,0],[0,-1]]).reshape([4,1,1,2])

def _get_Ab_hat():
    return np.array([1,0,0,0]).reshape([1,1,1,4])


def get_BPX_preconditioner_2D(L):
    dim = 2

    X_hat = _get_X_hat()
    Xb_hat = mode_product(X_hat, transpose_core(X_hat))

    U_hat = _get_U_hat()
    Ub_hat = mode_product(U_hat, transpose_core(U_hat))
    Ab_hat = np.array([1,0,0,0], float).reshape([1,1,1,4])
    Pb_hat = np.array([1,0,0,0], float).reshape([4,1,1,1])

    Ub = tensor_product(Ub_hat, Ub_hat)
    Xb = tensor_product(Xb_hat, Xb_hat)
    Ab = tensor_product(Ab_hat, Ab_hat)
    Pb = tensor_product(Pb_hat, Pb_hat)

    C_start = np.concatenate([Ab, Ab], axis=-1)
    C_end = np.concatenate([np.zeros_like(Pb), Pb], axis=0)

    C_factors = []
    for l in range(L):
        rank = 2**(2*dim+1)
        C_l = np.zeros([rank, 2**dim, 2**dim, rank])
        C_l[:rank//2, :, :, :rank//2] = Ub
        C_l[:rank//2, :, :, rank//2:] = (0.5**(l+1)) * Ub
        C_l[rank//2:, :, :, rank//2:] = (0.5 ** dim) * Xb
        C_factors.append(C_l)
    C = tm.TensorTrain([C_start] + C_factors + [C_end]).squeeze()
    return C


L = 4

U_hat = _get_U_hat()
Ub_hat = mode_product(U_hat, transpose_core(U_hat))
Ub = tensor_product(Ub_hat, Ub_hat)
Ab_hat = _get_Ab_hat()
Ab = tensor_product(Ab_hat, Ab_hat)

Z0_hat = mode_product(_get_Y0_hat(), transpose_core(_get_X_hat()))
Z1_hat = mode_product(_get_Y1_hat(), transpose_core(_get_X_hat()))
Z_10 = tensor_product(Z1_hat, Z0_hat)
W1_hat = _get_W1_hat()
W0_hat = _get_W0_hat()
W_10 = tensor_product(W1_hat, W0_hat)
K0_hat = mode_product(_get_N0_hat(), _get_P_hat())
K1_hat = mode_product(_get_N1_hat(), _get_P_hat())
K_10 = tensor_product(K1_hat, K0_hat)

# Build Q_2D = (M' x M) C
Qp_start = np.concatenate([Ab, strong_kronecker(Ab, W_10)], axis=-1)
Qp_end = np.concatenate([np.zeros([16, 2, 1, 1]), K_10], axis=0)
Qp_l = np.zeros([24, 4, 4, 24])
Qp_l[:16, :, :, :16] = Ub
Qp_l[:16, :, :, 16:] = strong_kronecker(Ub, W_10)
Qp_l[16:, :, :, 16:] = Z_10
Qp_2D = tm.TensorTrain([Qp_start] + [Qp_l.copy() for l in range(L)] + [Qp_end]).squeeze()

# Build Qp_1D = M' C
Qp_start = np.concatenate([Ab_hat, strong_kronecker(Ab_hat, W1_hat)], axis=-1)
Qp_end = np.concatenate([np.zeros([4, 1, 1, 1]), K1_hat], axis=0)
Qp_l = np.zeros([6, 2, 2, 6])
Qp_l[:4, :, :, :4] = Ub_hat
Qp_l[:4, :, :, 4:] = strong_kronecker(Ub_hat, W1_hat)
Qp_l[4:, :, :, 4:] = Z1_hat# * np.sqrt(2) # missing a factor of sqrt(2)
Qp_1D = tm.TensorTrain([Qp_start] + [Qp_l.copy() for _ in range(L)] + [Qp_end]).squeeze()

# Build Q_1D = M C
Q_start = np.concatenate([Ab_hat, strong_kronecker(Ab_hat, W0_hat)], axis=-1)
Q_end = np.concatenate([np.zeros([4, 2, 1, 1]), K0_hat], axis=0)
Ql_factors = []
for l in range(L):
    Q_l = np.zeros([8, 2, 2, 8])
    Q_l[:4, :, :, :4] = Ub_hat
    Q_l[:4, :, :, 4:] = strong_kronecker(Ub_hat, W0_hat)#* (0.5 ** (l+1))
    Q_l[4:, :, :, 4:] = Z0_hat# / np.sqrt(2)
    Ql_factors.append(Q_l / 2)
Q_1D = tm.TensorTrain([Q_start] + Ql_factors + [Q_end]).squeeze()

# Build Ql_1D = M Cl
l = 1
Ql_factors = [Ab_hat] + [Ub_hat.copy() for _ in range(l)] + [W0_hat] + [Z0_hat for _ in range(L-l)] + [K0_hat]
Ql = tm.TensorTrain(Ql_factors).squeeze() * 2**(-(L-l)) # missing a factor of 2**(0.5L)
Qpl_factors = [Ab_hat] + [Ub_hat.copy() for _ in range(l)] + [W1_hat] + [Z1_hat for _ in range(L-l)] + [K1_hat]
Qpl = tm.TensorTrain(Qpl_factors).squeeze() * 2**(L - (L-l)) # missing a factor of 2**(0.5L)
Ql_ref = get_bpx_Qt_term(L, l).transpose() * (2**l)
Qpl_ref = get_bpx_Qp_term(L, l) * (2**l)


C_2D = get_BPX_preconditioner_2D(L)
C_2D_ref = get_bpx_preconditioner_by_sum_2D(L)
#
#
# Build naive reference as (D x M) * C
C_expanded = C_2D.copy()
C_expanded.tensors.append(np.ones([1,1,1,1]))
Dx = get_derivative_matrix_as_tt(L)
Dx.tensors.append(np.ones([1,1,1,1]))
My = get_overlap_matrix_as_tt(L)
Qp_2D_ref = (Dx.copy().expand_dims([1,3]) * My.copy().expand_dims([0,2])).reshape_mode_indices([(4,4) for _ in range(L)] + [(2,1)])
Qp_2D_ref = Qp_2D_ref @ C_expanded

Qp_1D_ref = get_bpx_Qp(L)
Q_1D_ref = get_bpx_Qt(L).transpose()


def print_compare(A, B, labelA, labelB):
    nA = A.norm_squared()
    nB = B.norm_squared()
    nRes = (A-B).norm_squared()
    print(f"Norm² {labelA: <10}: {nA:8.3f}")
    print(f"Norm² {labelB: <10}: {nB:8.3f}")
    print(f"Norm² {'residual': <10}: {nRes:8.3f}")
    print(f"log2 {'ratio': <11}: {np.log2(nA/nB):8.3f}")
    print("-"*40)


# print_compare(Ql_ref, Ql, "Ql old", "Ql new")
# print_compare(Qpl_ref, Qpl, "Qpl old", "Qpl new")
# print_compare(Q_1D_ref, Q_1D, "Q_1D old", "Q_1D new")
# print_compare(Qp_1D_ref, Qp_1D, "Qp_1D old", "Qp_1D new")
print_compare(Qp_2D_ref, Qp_2D, "Qp_2D old", "Qp_2D new")



# print(f"1D: Norm Qpl old     : {Qpl_ref.norm_squared():8.3f}")
# print(f"1D: Norm Qpl         : {Qpl.norm_squared():8.3f}")
# print(f"1D: Norm residual  : {(Qpl-Qpl_ref).norm_squared():8.3f}")
# print("-"*40)
# #
# print(f"1D: Norm Q old     : {Q_1D_ref.norm_squared():8.3f}")
# print(f"1D: Norm Q         : {Q_1D.norm_squared():8.3f}")
# print(f"1D: Norm residual  : {(Q_1D-Q_1D_ref).norm_squared():8.3f}")
# print(f"1D: log2(norm)     : {np.log2(Q_1D.norm_squared()/Q_1D_ref.norm_squared()):8.3f}")
# print("-"*40)
#
# print(f"1D: Norm Q' old     : {Qp_1D_ref.norm_squared():8.3f}")
# print(f"1D: Norm Q'         : {Qp_1D.norm_squared():8.3f}")
# print(f"1D: Norm residual   : {(Qp_1D-Qp_1D_ref).norm_squared():8.3f}")
# print("-"*40)
# # print(f"2D: Norm Q naive   : {Q_2D_ref.norm_squared()}")
# # print(f"2D: Norm Q         : {Q_2D.norm_squared()}")
# # print(f"2D: Norm residual  : {(Q_2D-Q_2D_ref).norm_squared()}")
# #
# print(f"2D: Norm C old     : {C_2D_ref.norm_squared():8.3f}")
# print(f"2D: Norm C         : {C_2D.norm_squared():8.3f}")
# print(f"2D: Norm residual  : {(C_2D-C_2D_ref).norm_squared():8.3f}")
# print("-"*40)

#
#
# plt.close("all")
#
# C_ref_eval = C_ref.evalm()
# C_eval = C.evalm()
# residual = C_eval - C_ref_eval
# fig, axes = plt.subplots(2, 2, dpi=100, figsize=(14,9))
# axes[0][0].imshow(C_ref_eval)
# axes[0][0].set_title("Reference")
#
# axes[0][1].imshow(C_eval)
# axes[0][1].set_title("TT 2D C")
#
# axes[1][0].imshow(residual)
# axes[1][0].set_title("Residual")
#
# axes[1][1].imshow(np.log2(C_eval / C_ref_eval), clim=[-2, 2], cmap='bwr')
# axes[1][1].set_title("log2(naive/sum)")
