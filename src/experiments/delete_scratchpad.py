import numpy as np


A = np.random.normal(size=[1,2,3,4])
B = np.random.normal(size=[4,5,6,7])
shape_out = [A.shape[0], A.shape[1]*B.shape[1], A.shape[2]*B.shape[2], B.shape[3]]



C_td = np.tensordot(A, B, (-1, 0)).reshape(shape_out)
C_einsum = np.einsum("pabq,qABr->paAbBr", A, B).reshape(shape_out)
C_einsum2 = np.einsum("pabq,qABr->pabABr", A, B).reshape(shape_out)
C_loop = np.zeros(shape_out)
for i in range(A.shape[0]):
    for j in range(A.shape[-1]):
        for k in range(B.shape[-1]):
            C_loop[i, ..., k] += np.kron(A[i,...,j], B[j,...,k])


print("C_einsum == C_td", np.allclose(C_einsum, C_td))
print("C_einsum == C_einsum2", np.allclose(C_einsum, C_einsum2))
print("C_einsum == C_loop", np.allclose(C_einsum, C_loop))


