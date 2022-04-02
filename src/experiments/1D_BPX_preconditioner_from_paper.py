import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner, _get_bpx_factors, _get_refinement_tensor, get_bpx_preconditioner_by_sum_2D, get_derivative_matrix_as_tt, get_overlap_matrix_as_tt
import laplace_mps.tensormethods as tm

def mode_product(A,B):
    C = np.ones([A.shape[0] * B.shape[0],
                 A.shape[1], B.shape[2],
                 A.shape[-1]*B.shape[-1]]) * np.nan
    for i_a in range(A.shape[0]):
        for i_b in range(B.shape[0]):
            for j_a in range(A.shape[-1]):
                for j_b in range(B.shape[-1]):
                    C[i_a * B.shape[0] + i_b, ..., j_a * B.shape[-1] + j_b] = A[i_a, ..., j_a] @ B[i_b, ..., j_b]
    return C

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
    C = np.einsum("pabq,qABr->paAbBr", A, B)
    return C.reshape([C.shape[0], C.shape[1]*C.shape[2], C.shape[3]*C.shape[4], C.shape[5]])

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

    U_hat = _get_refinement_tensor()
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


L = 3

U_hat = _get_refinement_tensor()
Ub_hat = mode_product(U_hat, transpose_core(U_hat))
Ub = tensor_product(Ub_hat, Ub_hat)
Ab = tensor_product(_get_Ab_hat(), _get_Ab_hat())

Z0_hat = mode_product(_get_Y0_hat(), transpose_core(_get_X_hat()))
Z1_hat = mode_product(_get_Y1_hat(), transpose_core(_get_X_hat()))
Z = tensor_product(Z1_hat, Z0_hat)
W = tensor_product(_get_W1_hat(), _get_W0_hat())
K0_hat = mode_product(_get_N0_hat(), _get_P_hat())
K1_hat = mode_product(_get_N1_hat(), _get_P_hat())
K = tensor_product(K1_hat, K0_hat)

Q_start = np.concatenate([Ab, strong_kronecker(Ab, W)], axis=-1)
Q_end = np.concatenate([np.zeros([16, 2, 1, 1]), K], axis=0)

Q_l = np.zeros([24, 4, 4, 24])
Q_l[:16, :, :, :16] = Ub
Q_l[:16, :, :, 16:] = strong_kronecker(Ub, W)
Q_l[16:, :, :, 16:] = Z
Q_2D = tm.TensorTrain([Q_start] + [Q_l for _ in range(L)] + [Q_end]).squeeze()


C = get_BPX_preconditioner_2D(L)
C_expanded = C.copy()
C_expanded.tensors.append(np.ones([1,1,1,1]))
C_ref = get_bpx_preconditioner_by_sum_2D(L)

Dx = get_derivative_matrix_as_tt(L)
Dx.tensors.append(np.ones([1,1,1,1]))
My = get_overlap_matrix_as_tt(L)
Q_2D_ref = (Dx.expand_dims([1,3]) * My.expand_dims([0,2])).reshape_mode_indices([(4,4) for _ in range(L)] + [(2,1)])
Q_2D_ref = Q_2D_ref @ C_expanded

print(f"Norm Q naive   : {Q_2D_ref.norm_squared()}")
print(f"Norm Q         : {Q_2D.norm_squared()}")
print(f"Norm residual  : {(Q_2D-Q_2D_ref).norm_squared()}")





plt.close("all")

C_ref_eval = C_ref.evalm()
C_eval = C.evalm()
residual = C_eval - C_ref_eval
fig, axes = plt.subplots(2, 2, dpi=100, figsize=(14,9))
axes[0][0].imshow(C_ref_eval)
axes[0][0].set_title("Reference")

axes[0][1].imshow(C_eval)
axes[0][1].set_title("TT 2D C")

axes[1][0].imshow(residual)
axes[1][0].set_title("Residual")

axes[1][1].imshow(np.log2(C_eval / C_ref_eval), clim=[-2, 2], cmap='bwr')
axes[1][1].set_title("log2(naive/sum)")
