import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner, _get_bpx_factors, _get_refinement_tensor
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


L = 3
C_julia = get_bpx_preconditioner(L)

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
U_bpx_factor, V, W_bpx_factor, Y = _get_bpx_factors()

C_start = np.array([1,0,0,0, 1,0,0,0], float).reshape([1, 1, 1, 8])
C_end = np.array([0,0,0,0, 1,0,0,0], float).reshape([8,1,1,1])

C_factors = []
for l in range(L):
    C_l = np.zeros([8, 2, 2, 8])
    C_l[:4, :, :, :4] = Ub_hat
    C_l[:4, :, :, 4:] = (0.5**(l+1)) * Ub_hat
    C_l[4:, :, :, 4:] = (0.5 ** 1) * Xb_hat
    C_factors.append(C_l)
C = tm.TensorTrain([C_start] + C_factors + [C_end]).squeeze()


C_julia_eval = C_julia.evalm()
C_eval = C.evalm()
residual = C_eval - C_julia_eval
fig, axes = plt.subplots(2, 2, dpi=100, figsize=(14,9))
axes[0][0].imshow(C_julia_eval)
axes[0][0].set_title("Naive C (as sum)")

axes[0][1].imshow(C_eval)
axes[0][1].set_title("TT 2D C")

axes[1][0].imshow(residual)
axes[1][0].set_title("Residual")

axes[1][1].imshow(np.log2(C_eval / C_julia_eval), clim=[-2, 2], cmap='bwr')
axes[1][1].set_title("log2(naive/sum)")


#%%

# plt.close("all")
# fig, axes = plt.subplots(2,2,figsize=(14,8), dpi=100)
# axes[0][0].imshow(U_bpx_factor.transpose([0,1,3,2]).reshape([8,8]))
# axes[0][0].set_title("U from julia code")
# axes[0][1].imshow(W_bpx_factor.transpose([0,1,3,2]).reshape([8,8]))
# axes[0][1].set_title("W from julia code")
#
# axes[1][0].imshow(Ub_hat.transpose([0,1,3,2]).reshape([8,8]))
# axes[1][0].set_title("Ub_hat")
# axes[1][1].imshow(Xb_hat.transpose([0,1,3,2]).reshape([8,8]))
# axes[1][1].set_title("Xb_hat")
#
#
# for ax in axes:
#     ax.axhline(3.5, color='w')
#     ax.axvline(3.5, color='w')
