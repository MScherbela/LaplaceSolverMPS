import numpy
import numpy as np
import laplace_mps.tensormethods as tm

def _get_legendre_zoom_in_tensor(degree):
    n = degree + 1
    tensor = np.zeros([n,2,n])
    for i in range(n):
        coeff = np.zeros(n)
        coeff[i] = 1
        p = numpy.polynomial.legendre.Legendre(coeff)
        p_left = p.convert(domain=[-1, 0], kind=numpy.polynomial.legendre.Legendre, window=[-1, 1])
        p_right = p.convert(domain=[0, 1], kind=numpy.polynomial.legendre.Legendre, window=[-1, 1])
        tensor[i,0,:len(p_left.coef)] = p_left.coef
        tensor[i,1,:len(p_right.coef)] = p_right.coef
    return tensor


def _polynomial_to_legendre(coeffs):
    """Convert a coefficients of an algebraic polynomial of degree 2 and calculate the corresponding coefficients of a Legendre polynomial"""
    return np.array([[1, 1/2, 1/3],
                     [0, 1/2, 1/2],
                     [0, 0, 1/6]]) @ np.array(coeffs)


def _get_gram_matrix_legendre():
    return np.diag([2,2/3])

def get_rhs_matrix_as_tt(L):
    M = get_overlap_matrix_as_tt(L)
    R = M.transpose()

    G = _get_gram_matrix_legendre() / 2
    R.tensors[-1] = (R.tensors[-1].squeeze(-1) @ G)[...,None]
    return R


def get_polynomial_as_tt(coeffs, L):
    legendre_coeffs = numpy.polynomial.Polynomial(coeffs).convert(kind=numpy.polynomial.legendre.Legendre, domain=[0,1]).coef
    U_l = _get_legendre_zoom_in_tensor(len(coeffs)-1)

    poly_value_left = np.ones(len(coeffs))
    poly_value_left[1::2] *= -1
    poly_values_right = np.ones(len(coeffs))

    U_first = legendre_coeffs[None,:]
    U_last = np.array([poly_values_right + poly_value_left, poly_values_right - poly_value_left]).T[...,None] / 2

    tensors = [U_l for _ in range(L)]
    tensors[0] = np.tensordot(U_first, tensors[0], axes=[-1,0])
    tensors.append(U_last)
    return tm.TensorTrain(tensors)


def evaluate_nodal_basis(tt: tm.TensorTrain, s: np.array):
    s = np.array(s)
    assert tt[-1].shape[1] == 2, "Tensor to be evaluated must be in nodal-basis, i.e. have 2 basis elements"
    output = tm.TensorTrain(tt)
    final_factor = np.array([np.ones_like(s), s])
    output.tensors[-1] = (output.tensors[-1].squeeze(-1) @ final_factor)[..., None]
    return output

def evaluate_nodal_basis(tt: tm.TensorTrain, s: np.array):
    s = np.array(s)
    assert tt[-1].shape[1] == 2, "Tensor to be evaluated must be in nodal-basis, i.e. have 2 basis elements"
    output = tm.TensorTrain(tt)
    final_factor = np.array([np.ones_like(s), s])
    output.tensors[-1] = (output.tensors[-1].squeeze(-1) @ final_factor)[..., None]
    return output

def hat_to_nodal_basis(tt: tm.TensorTrain):
    L = len(tt)
    M = get_overlap_matrix_as_tt(L)
    tt = tt.copy()
    tt.tensors.append(np.eye(2).reshape([1,2,2,1]))
    return M @ tt



def _get_refinement_tensor():
    """Returns tensor of shape [2, 2,2, 2]: [[I, J.T], [0, J]]
    This tensor forms the core of the tensor-trains for the overlap- and derivative-matrix"""
    J = np.array([[0, 1], [0, 0]], dtype=float)
    M = np.zeros([2, 2, 2, 2])
    M[0, :, :, 0] = np.eye(2)
    M[0, :, :, 1] = J.T
    M[1, :, :, 1] = J
    return M

def get_overlap_matrix_as_tt(L):
    """Retuns a tensor-train representing the matrix ':math:`M_ij = \int phi_i phi_j` where phi_i and phi_j are hat-like basis functions"""
    M = _get_refinement_tensor()
    M_first = np.array([1, 0]).reshape([1, 2])
    M_last = np.array([[1,1],[1,-1]]).reshape(2, 2, 1) / 2
    tensors = [M for _ in range(L)]
    tensors[0] = np.tensordot(M_first,tensors[0], axes=[-1,0])
    tensors.append(M_last)
    return tm.TensorTrain(tensors)

def get_derivative_matrix_as_tt(L):
    """Returns a tensor-train representing the matris M' which computes the first derivatives of a function in a hat-like basis."""
    M = _get_refinement_tensor() * 2
    M_first = np.array([1, 0]).reshape([1, 2])
    M_last = np.array([1,-1]).reshape(2, 1)
    tensors = [M for _ in range(L)]
    tensors[0] = np.tensordot(M_first, tensors[0], axes=[-1, 0])
    tensors[-1] = np.tensordot(tensors[-1], M_last, axes=[-1, 0])
    return tm.TensorTrain(tensors)

def get_laplace_matrix_as_tt(L):
    Mp = get_derivative_matrix_as_tt(L)
    A = Mp.copy().transpose() @ Mp
    return A.reapprox(rel_error=1e-16)


def get_level_mapping_matrix_as_tt(L,l):
    M = _get_refinement_tensor()
    A = np.zeros([2, 2, 2])
    A[0, :, 0] = [0.5, 1]
    A[0, :, 1] = [0, 0.5]
    A[1, :, 0] = [0.5, 0]
    A[1, :, 1] = [1, 0.5]
    start = np.array([1, 0])[None, None, :]
    end = np.array([1, 0])[:, None, None]
    return tm.TensorTrain([start] + [M] * l + [A] * (L - l) + [end]).squeeze()

def getBPXPreconditioner(L):
    P0 = get_level_mapping_matrix_as_tt(L, 0)
    C = P0 @ P0.copy().transpose()
    for l in range(1,L):
        P = get_level_mapping_matrix_as_tt(L, l)
        C = C + 2**(-l) * P @ P.copy().transpose()
        C.reapprox(rel_error=1e-16)
    return C

def solve_PDE_1D(f, **solver_options):
    L = len(f) - 1
    g = (get_rhs_matrix_as_tt(L) @ f).squeeze()
    A = get_laplace_matrix_as_tt(L)
    return solve_with_grad_descent(A, g, **solver_options)

def solve_with_grad_descent(A, b, n_steps=200, lr=1.0, max_rank=20, print_steps=False, recalc_residual_every_n=100):
    x = tm.zeros(b.mode_sizes)
    for n in range(n_steps):
        if n%recalc_residual_every_n == 0:
            r = b - A @ x
        Ar = A @ r
        r2 = float((r @ r).eval().flatten())
        gamma = lr * r2 / float((r @ Ar).eval().flatten())

        x = x + gamma * r
        r = r - gamma * Ar
        x.reapprox(ranks_new=max_rank)
        r.reapprox(ranks_new=max_rank)
        if print_steps and (n%50 == 0):
            print(f"Step {n:4d}: ||r||² = {r2:.2e}, gamma = {gamma: .2e}")
    return x

#
# if __name__ == '__main__':
#     C = getBPXPreconditioner(5)







