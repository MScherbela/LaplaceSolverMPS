import numpy as np
import laplace_mps.tensormethods as tm
from laplace_mps.utils import kronecker_prod_2D, _get_gram_matrix_tt


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

def _get_Pb_hat():
    return np.array([1,0,0,0], float).reshape([4,1,1,1])

def _get_W0_hat():
    return np.array([[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]).reshape([4,1,1,4])

def _get_W1_hat():
    return np.array([[1,0],[0,1],[-1,0],[0,-1]]).reshape([4,1,1,2])

def _get_Ab_hat():
    return np.array([1,0,0,0]).reshape([1,1,1,4])

def _get_Ub_hat():
    U_hat = _get_U_hat()
    return mode_product(U_hat, transpose_core(U_hat))

def _get_Z0_hat():
    return mode_product(_get_Y0_hat(), transpose_core(_get_X_hat()))

def _get_Z1_hat():
    return mode_product(_get_Y1_hat(), transpose_core(_get_X_hat()))

def _get_K0_hat():
    return mode_product(_get_N0_hat(), _get_P_hat())

def _get_K1_hat():
    return mode_product(_get_N1_hat(), _get_P_hat())


def get_BPX_preconditioner_2D(L):
    X_hat = _get_X_hat()
    Xb_hat = mode_product(X_hat, transpose_core(X_hat))
    Ub = tensor_product(_get_Ub_hat(), _get_Ub_hat())
    Xb = tensor_product(Xb_hat, Xb_hat)
    Ab = tensor_product(_get_Ab_hat(), _get_Ab_hat())
    Pb = tensor_product(_get_Pb_hat(), _get_Pb_hat())

    C_start = np.concatenate([Ab, Ab], axis=-1)
    C_end = np.concatenate([np.zeros_like(Pb), Pb], axis=0)

    C_factors = []
    for l in range(L):
        rank = 2**(4+1)
        C_l = np.zeros([rank, 4, 4, rank])
        C_l[:rank//2, :, :, :rank//2] = Ub
        C_l[:rank//2, :, :, rank//2:] = (0.5**(l+1)) * Ub
        C_l[rank//2:, :, :, rank//2:] = (0.5 ** 2) * Xb
        C_factors.append(C_l)
    C = tm.TensorTrain([C_start] + C_factors + [C_end]).squeeze()
    return C

def get_BPX_Qp_2D(L, deriv='x'):
    Ub = tensor_product(_get_Ub_hat(), _get_Ub_hat())
    Ab = tensor_product(_get_Ab_hat(), _get_Ab_hat())
    W_10 = tensor_product(_get_W1_hat(), _get_W0_hat())
    Z_10 = tensor_product(_get_Z1_hat(), _get_Z0_hat())
    K_10 = tensor_product(_get_K1_hat(), _get_K0_hat())
    Qp_start = np.concatenate([Ab, strong_kronecker(Ab, W_10)], axis=-1)
    Qp_end = np.concatenate([np.zeros([16, 2, 1, 1]), K_10], axis=0)
    Q_factors = []
    for l in range(L):
        Qp_l = np.zeros([24, 4, 4, 24])
        Qp_l[:16, :, :, :16] = Ub
        Qp_l[:16, :, :, 16:] = strong_kronecker(Ub, W_10)
        Qp_l[16:, :, :, 16:] = Z_10 / 2
        Q_factors.append(Qp_l)
    Qp_2D = tm.TensorTrain([Qp_start] + Q_factors + [Qp_end]).squeeze()
    if deriv == 'x':
        pass
    elif deriv == 'y':
        Qp_2D.reshape_mode_indices([(2, 2, 2, 2) for _ in range(L)] + [(1, 2, 1, 1)])
        for i, U in enumerate(Qp_2D):
            Qp_2D.tensors[i] = np.transpose(U, [0, 2, 1, 4, 3, 5])
        Qp_2D.reshape_mode_indices([(4, 4) for _ in range(L)] + [(2, 1)])
    else:
        raise ValueError("Unknown axis")
    return Qp_2D

def get_BPX_Q_2D(L):
    Ub = tensor_product(_get_Ub_hat(), _get_Ub_hat())
    Ab = tensor_product(_get_Ab_hat(), _get_Ab_hat())
    W_00 = tensor_product(_get_W0_hat(), _get_W0_hat())
    Z_00 = tensor_product(_get_Z0_hat(), _get_Z0_hat())
    K_00 = tensor_product(_get_K0_hat(), _get_K0_hat())
    Q_start = np.concatenate([Ab, strong_kronecker(Ab, W_00)], axis=-1)
    Q_end = np.concatenate([np.zeros([16, 4, 1, 1]), K_00], axis=0)
    Q_factors = []
    for l in range(L):
        Q_l = np.zeros([32, 4, 4, 32])
        Q_l[:16, :, :, :16] = Ub
        Q_l[:16, :, :, 16:] = strong_kronecker(Ub, W_00)  # * 0.5**l
        Q_l[16:, :, :, 16:] = Z_00 / 2
        Q_factors.append(Q_l / 2)
    Q_2D = tm.TensorTrain([Q_start] + Q_factors + [Q_end]).squeeze()
    return Q_2D


def get_laplace_BPX_2D(L):
    DxMyC = get_BPX_Qp_2D(L, 'x')
    DyMxC = get_BPX_Qp_2D(L, 'y')

    I = tm.TensorTrain([np.eye(2)[None, :, :, None] for _ in range(L)] + [np.eye(1)[None, :, :, None]])
    G = _get_gram_matrix_tt(L) * 2
    Gy = kronecker_prod_2D(I, G)
    Gx = kronecker_prod_2D(G, I)

    B = DxMyC.copy().transpose() @ Gy @ DxMyC + DyMxC.copy().transpose() @ Gx @ DyMxC
    for i,U in enumerate(B):
        B.tensors[i] = U / 4
    return B.squeeze()

def get_rhs_matrix_BPX_2D(L):
    Qt = get_BPX_Q_2D(L).transpose()
    gram_matrix = np.diag([2,2/3]) @ np.array([[1, 1], [-1, 1]]) * 0.5

    G = tm.TensorTrain([np.eye(2)[None, :, :, None] for _ in range(L)] + [gram_matrix.reshape([1,2,2,1])])
    G = kronecker_prod_2D(G, G)
    R = Qt @ G
    for i, U in enumerate(R):
        R.tensors[i] = U / 4
    return R



if __name__ == '__main__':
    L = 8
    B = get_laplace_BPX_2D(L)
    B.reapprox(ranks_new=400, rel_error=1e-15)
    print(B.shapes)
