import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner, _get_bpx_factors, _get_refinement_tensor, get_bpx_preconditioner_by_sum_2D
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

L = 5
dim = 2

X_hat = np.zeros([2,2,2])
X_hat[0,:,0] = [1,2]
X_hat[0,:,1] = [0,1]
X_hat[1,:,0] = [1,0]
X_hat[1,:,1] = [2,1]
X_hat = X_hat[:,:, None, :] / 2
X_hat_transp = transpose_core(X_hat)
Xb_hat = mode_product(X_hat, X_hat_transp)

U_hat = _get_refinement_tensor()
U_hat_transp = transpose_core(U_hat)
Ub_hat = mode_product(U_hat, U_hat_transp)
Ab_hat = np.array([1,0,0,0], float).reshape([1,1,1,4])
Pb_hat = np.array([1,0,0,0], float).reshape([4,1,1,1])

if dim == 1:
    Ub = Ub_hat
    Xb = Xb_hat
    Ab = Ab_hat
    Pb = Pb_hat
elif dim == 2:
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

if dim == 1:
    C_ref = get_bpx_preconditioner(L)
elif dim == 2:
    C_ref = get_bpx_preconditioner_by_sum_2D(L)

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
