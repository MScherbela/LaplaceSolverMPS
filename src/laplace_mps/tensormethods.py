import numpy as np
from typing import List
from scipy.sparse.linalg import svds

def _get_required_ranks_for_accuracy(s, eps):
    cum_error = s[::-1] ** 2
    # if cum_error[-1] == 0:
    #     # cumulative error is 0, even when dropping everything
    #     return 1
    cum_error /= cum_error[-1]
    n_to_remove = np.sum(cum_error < eps ** 2)
    return len(s) - n_to_remove

def zeros(mode_sizes):
    return TensorTrain([np.zeros((1,) + tuple(m) + (1,)) for m in mode_sizes])

class TensorTrain:
    def __len__(self):
        return len(self.tensors)

    def __init__(self, tensors):
        if isinstance(tensors, TensorTrain):
            self.tensors = [U.copy() for U in tensors]
        else:
            self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    @property
    def ranks(self):
        return [U.shape[-1] for U in self.tensors[:-1]]

    @property
    def shapes(self):
        return [U.shape for U in self.tensors]

    @property
    def mode_sizes(self):
        return [U.shape[1:-1] for U in self.tensors]

    @classmethod
    def from_ttpylist(cls, tensors):
        tensors = [U.transpose(np.arange(U.ndim)[::-1]) for U in tensors][::-1]
        return cls(tensors)

    def to_ttpylist(self):
        return [U.transpose(np.arange(U.ndim)[::-1]) for U in self.tensors][::-1]


    @classmethod
    def ttsvd(cls, A: np.array, ranks: List[int], use_sparse_svd=False):
        d = len(A.shape)
        assert len(ranks) == (d - 1)
        ranks = [1] + list(ranks) + [1]

        tensors = []
        right_rest = A
        singular_values = []
        for k in range(d-1):
            B = np.reshape(right_rest, [ranks[k]*A.shape[k], -1])
            if use_sparse_svd:
                U,D,Vt = svds(B, ranks[k+1])
            else:
                U,D,Vt = np.linalg.svd(B, full_matrices=False)
                singular_values.append(D.copy())
                U = U[:, :ranks[k+1]]
                D = D[:ranks[k+1]]
                Vt = Vt[:ranks[k+1], :]

            U_shape = [ranks[k], A.shape[k], ranks[k+1]]
            tensors.append(U.reshape(U_shape))
            right_rest = D[:, None] * Vt
        tensors.append(right_rest.reshape([ranks[-2], -1, ranks[-1]]))
        return cls(tensors)

    def copy(self):
        return TensorTrain([U.copy() for U in self.tensors])


    def right_orthogonalize(self):
        U = self.tensors[0]
        for k, r in enumerate(self.ranks):
            Q,R = np.linalg.qr(U.reshape([-1, r]))
            self.tensors[k] = Q.reshape(U.shape)
            U = np.tensordot(R, self.tensors[k+1], [1, 0])
        self.tensors[-1] = U
        return self

    def left_orthogonalize(self):
        U = self.tensors[-1]
        for k in reversed(range(1, len(self))):
            r = U.shape[0]
            Qt,Rt = np.linalg.qr(U.reshape([r, -1]).T)
            orth_factor = (Qt.T).reshape((Qt.shape[1],) + U.shape[1:])
            self.tensors[k] = orth_factor
            U = np.tensordot(self.tensors[k-1], Rt.T, [-1, 0])
        self.tensors[0] = U
        return self

    def reapprox(self, ranks_new=None, rel_error=1e-16, orthogonalize=True):
        if orthogonalize:
            self.left_orthogonalize()

        if ranks_new is None:
            ranks_new = self.ranks
        if isinstance(ranks_new, int):
            ranks_new = [ranks_new] * (len(self) - 1)

        U = self.tensors[0]
        for k, (r_old, r_new) in enumerate(zip(self.ranks, ranks_new)):
            U_reshaped = U.reshape([-1, r_old])
            # Run SVD and truncate singular values
            if (r_new < r_old) and r_new < min(U_reshaped.shape):
                Vl, S, Vr = svds(U_reshaped, k=r_new)
                # SVDs sorts singular values from smallest to largest => flip order
                Vl = Vl[:, ::-1]
                S = S[::-1]
                Vr = Vr[::-1, :]
            else:
                Vl,S,Vr = np.linalg.svd(U_reshaped, full_matrices=False)
            r_new = min(U_reshaped.shape[0], r_old, r_new, _get_required_ranks_for_accuracy(S, rel_error))
            Vl = Vl[:, :r_new]
            R = S[:r_new, None] * Vr[:r_new,:]

            # Store left singular values (after appropriate reshaping)
            Vl = Vl.reshape(U.shape[:-1] + (r_new,))
            self.tensors[k] = Vl

            # push right singular values/vectors onto next factor on the right
            U = np.tensordot(R, self.tensors[k+1], [1, 0])
        self.tensors[-1] = U
        return self

    def __add__(self, other):
        assert len(other) == len(self)
        output_tensors = []
        for k, (U,V) in enumerate(zip(self, other)):
            if k == 0:
                output_tensors.append(np.concatenate([U,V], axis=-1))
            elif k == (len(self)-1):
                output_tensors.append(np.concatenate([U,V], axis=0))
            else:
                mode_shape = U.shape[1:-1]
                zeros_10 = np.zeros(V.shape[:1] + mode_shape + U.shape[-1:])
                zeros_01 = np.zeros(U.shape[:1] + mode_shape + V.shape[-1:])
                W = np.concatenate([np.concatenate([U, zeros_01], axis=-1), np.concatenate([zeros_10, V], axis=-1)], axis=0)
                output_tensors.append(W)
        return TensorTrain(output_tensors)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        """ TensorTrain() x other"""
        if isinstance(other, (float, int)):
            output = self.copy()
            output.tensors[0] *= other
            return output

        assert len(other) == len(self)
        output_tensors = []
        for k, (U,V) in enumerate(zip(self, other)):
            W = np.einsum('p...q,P...Q->pP...qQ', U,V, optimize=True)
            new_shape = (W.shape[0] * W.shape[1],) + W.shape[2:-2] + (W.shape[-2] * W.shape[-1],)
            output_tensors.append(W.reshape(new_shape))
        return TensorTrain(output_tensors)

    def __rmul__(self, other):
        """scalar x TensorTrain()"""
        assert isinstance(other, (float, int, np.float)), "Only defined for scalar multiplication"
        output = self.copy()
        output.tensors[0] *= other
        return output

    def __matmul__(self, other):
        assert len(other) == len(self)
        output_tensors = []
        for k, (U,V) in enumerate(zip(self, other)):
            W = np.tensordot(U, V, axes=[-2, 1])
            if U.ndim == 4 and V.ndim == 4:
                W = W.transpose([0, 3, 1, 4, 2, -1])
            elif U.ndim == 3 and V.ndim == 4:
                W = W.transpose([0, 2, 3, 1, -1])
            elif U.ndim == 4 and V.ndim == 3:
                W = W.transpose([0, 3, 1, 2, -1])
            elif U.ndim == 3 and V.ndim == 3:
                W = W.transpose([0, 2, 1, -1])
            else:
                raise NotImplementedError("Currently only supports 1D or 2D mode dimensions")

            W = W.reshape((W.shape[0] * W.shape[1],) + W.shape[2:-2] + (W.shape[-2] * W.shape[-1],))
            output_tensors.append(W)
        return TensorTrain(output_tensors)


    def __neg__(self):
        t = self.copy()
        t.tensors[0] *= -1
        return t

    def transpose(self, perm=None):
        perm = perm or [1,0]
        perm = [0] + [p+1 for p in perm] + [-1]
        for k,U in enumerate(self.tensors):
            if U.ndim > 3:
                self.tensors[k] = np.transpose(U, perm)
        return self

    def flatten_mode_indices(self):
        for k,U in enumerate(self.tensors):
            self.tensors[k] = U.reshape([U.shape[0], -1, U.shape[-1]])
        return self

    def unflatten_mode_indices(self, mode_shapes):
        for k, (U,s) in enumerate(zip(self.tensors, mode_shapes)):
            self.tensors[k] = U.reshape(U.shape[:1] + tuple(s) + U.shape[-1:])
        return self

    def reshape_mode_indices(self, mode_shapes: List[List[int]]):
        if isinstance(mode_shapes[0], int):
            mode_shapes = [mode_shapes] * len(self)
        for i,U in enumerate(self.tensors):
            self.tensors[i] = U.reshape((U.shape[0],) + tuple(mode_shapes[i]) + (U.shape[-1],))
        return self

    def evalm(self):
        return self.eval(reshape='matrix', squeeze=False)

    def evalv(self):
        return self.eval(reshape='vector', squeeze=False)

    def eval(self, squeeze=True, reshape=None):
        A = self.tensors[0]
        for U in self.tensors[1:]:
            A = np.tensordot(A, U, (-1, 0))
        A = A.reshape(A.shape[1:-1])
        if squeeze:
            A = A.squeeze()
        if reshape == 'vector':
            return A.flatten()
        elif reshape == 'matrix':
            L = len(self)
            row_indices = list(range(0,2*L,2))
            col_indices = list(range(1,2*L,2))
            A = A.transpose(row_indices + col_indices)
            return A.reshape([np.prod(A.shape[:L]), -1])
        else:
            return A

    def squeeze(self):
        new_tensors = [self.tensors[0]]
        for mode_size, U in zip(self.mode_sizes[1:], self.tensors[1:]):
            rightmost_u = new_tensors[-1]
            if np.prod(rightmost_u.shape[1:-1]) == 1: # right-most U in new tensors has mode-size 1
                rightmost_u = rightmost_u.reshape([rightmost_u.shape[0], rightmost_u.shape[-1]])
                new_tensors[-1] = np.tensordot(rightmost_u, U, axes=[-1, 0])
            elif np.prod(mode_size) == 1: # tensor to be added has mode size 1
                U = U.reshape([U.shape[0], U.shape[-1]]) # squeeze away mode indices
                new_tensors[-1] = np.tensordot(rightmost_u, U, axes=[-1,0])
            else:
                new_tensors.append(U)
        self.tensors = new_tensors
        return self

    def expand_dims(self, axis=-1):
        for k, U in enumerate(self.tensors):
            if isinstance(axis, int):
                pos = axis - 1 if axis < 0 else axis + 1
            else:
                pos = [a - 1 if a < 0 else a + 1 for a in axis]
            self.tensors[k] = np.expand_dims(U, pos)
        return self
    #
    # def expand_ranks(self, mode_indices):
    #     for n in range(n_end):
    #         self.tensors.append(np.ones([1,1,1]))
    #     return self

    def norm_squared(self, orthogonalize=True):
        A = self.copy()
        A.left_orthogonalize()
        return float(np.sum(A[0]**2))


if __name__ == '__main__':
    np.random.seed(0)
    A = np.random.normal(size=[2,2,2,2,2])
    A = TensorTrain.ttsvd(A, ranks=[2]*4)
    norm_direct = (A @ A).squeeze().eval()
    A_orth = A.left_orthogonalize()

    print(norm_direct)

