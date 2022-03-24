import matplotlib.pyplot as plt
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

def get_rhs_matrix_as_tt(L, basis="corner"):
    R = get_overlap_matrix_as_tt(L).transpose() * 0.5

    G = _get_gram_matrix_legendre() / 2
    if basis == 'corner':
        G = G @  np.array([[1,1],[-1,1]]) # convert from left/right-basis to Legendre (P0, P1) basis
    elif basis == "legendre":
        pass
    else:
        raise NotImplementedError(f"Unknown basis {basis}")

    R.tensors[-1] = (R.tensors[-1].squeeze(-1) @ G)[...,None]
    return R

def get_rhs_matrix_as_tt_2D(L):
    R = get_rhs_matrix_as_tt(L).squeeze()
    Rx = R.copy().expand_dims([1,3])
    Rx.tensors[-1] = Rx.tensors[-1].squeeze(axis=-2)
    Ry = R.copy().expand_dims([0,2])
    Ry.tensors[-1] = Ry.tensors[-1].squeeze(axis=-2)
    R = (Rx * Ry).reshape_mode_indices([[4,4]]*L + [[2,2]]).reapprox(rel_error=1e-15)
    R.tensors[-1] = R.tensors[-1].reshape([-1, 4, 1])
    return R


def to_legendre_basis(t: tm.TensorTrain):
    t = t.copy()
    U = np.array([[1, -1], [1, 1]]) / 2  # convert from left/right-basis to Legendre (P0, P1) basis
    t.tensors[-1] = (t.tensors[-1].squeeze(-1) @ U)[...,None]
    return t


def get_polynomial_as_tt(coeffs, L):
    """
    coeffs are the polynomial coeffs, starting with the x^0 coeff
    """
    legendre_coeffs = numpy.polynomial.Polynomial(coeffs).convert(kind=numpy.polynomial.legendre.Legendre, domain=[0,1]).coef
    U_l = _get_legendre_zoom_in_tensor(len(coeffs)-1)

    poly_value_left = np.ones(len(coeffs))
    poly_value_left[1::2] *= -1
    poly_values_right = np.ones(len(coeffs))

    U_first = legendre_coeffs[None,:]
    U_last = np.array([poly_value_left, poly_values_right]).T[...,None]

    tensors = [U_l for _ in range(L)]
    tensors[0] = np.tensordot(U_first, tensors[0], axes=[-1,0])
    tensors.append(U_last)
    return tm.TensorTrain(tensors)


def get_trig_function_as_tt(coeffs, L):
    a,b,nu = coeffs
    s,c = np.sin(nu * np.pi), np.cos(nu * np.pi)
    C_0 = np.array([[c,-s],[s,c]])
    C_0 = np.array([a,b]) @ C_0

    C_tensors = [C_0.reshape([1,1,-1])]
    for l in range(1, L+1):
        C_l = []
        for i in range(2):
            phase = 0.5**l * np.pi * nu * (2*i-1)
            s, c = np.sin(phase), np.cos(phase)
            C_l.append(np.array([[c,-s],[s,c]]))
        C_l = np.stack(C_l, axis=1)
        C_tensors.append(C_l)
    C_final = np.array([[c, c],[-s,s]])[...,None]
    C_tensors.append(C_final)
    return tm.TensorTrain(C_tensors).squeeze()


def evaluate_nodal_basis(tt: tm.TensorTrain, s: np.array, basis='corner'):
    s = np.array(s)
    output = tt.copy()
    if s.ndim == 1:
        # Approximation of 1D function
        assert tt[-1].shape[1] == 2, "Tensor to be evaluated must be in nodal-basis, i.e. have 2 basis elements"
        if basis == 'corner':
            final_factor = np.array([(1-s)/2, (1+s)/2])
        elif basis == 'legendre':
            final_factor = np.array([np.ones_like(s), 2*s - 1])
        else:
            raise NotImplementedError(f"Unknown basis {basis}")
        output.tensors[-1] = (output.tensors[-1].squeeze(-1) @ final_factor)[..., None]

    elif (s.ndim == 2) and s.shape[0] == 2:
        # Approximation of 2D function
        assert tt[-1].shape[1:3] == (2,2)
        if basis == 'corner':
            final_factor = np.array([(1-s[0])*(1-s[1]), (1-s[0])*(1+s[1]), (1+s[0])*(1-s[1]), (1+s[0])*(1+s[1])]) / 4
            output.tensors[-1] = (output.tensors[-1].reshape([-1, 4]) @ final_factor)[..., None]
        else:
            raise NotImplementedError(f"Unknown basis {basis}")
    return output


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
    M = _get_refinement_tensor() * np.sqrt(2)
    M_first = np.array([1, 0]).reshape([1, 2])
    M_last = np.array([1,-1]).reshape(2, 1)
    tensors = [M for _ in range(L)]
    tensors[0] = np.tensordot(M_first, tensors[0], axes=[-1, 0])
    tensors[-1] = np.tensordot(tensors[-1], M_last, axes=[-1, 0])
    return tm.TensorTrain(tensors)

def get_laplace_matrix_as_tt(L):
    Mp = get_derivative_matrix_as_tt(L)
    A = Mp.copy().transpose() @ Mp
    A.reapprox(rel_error=1e-16)
    # for i in range(L):
    #     A.tensors[i] *= 2
    return A

def get_level_mapping_matrix_as_tt(L,l):
    M = _get_refinement_tensor()
    A = np.zeros([2, 2, 1, 2])
    A[0, :, 0, 0] = [0.5, 1]
    A[0, :, 0, 1] = [0, 0.5]
    A[1, :, 0, 0] = [0.5, 0]
    A[1, :, 0, 1] = [1, 0.5]
    A /= np.sqrt(2)
    start = np.array([1, 0])[None, None, :]
    end = np.array([1, 0])[:, None, None]
    return tm.TensorTrain([start] + [M] * l + [A] * (L - l) + [end]).squeeze()

def get_bpx_preconditioner_by_sum(L):
    P0 = get_level_mapping_matrix_as_tt(L, 0)
    C = P0 @ P0.copy().transpose()
    for l in range(1,L+1):
        P = get_level_mapping_matrix_as_tt(L, l)
        PP = P @ P.copy().transpose()
        C = (C + 2**(-l) * PP).reapprox(ranks_new=8)
    return C

def get_bpx_preconditioner_by_sum_2D(L):
    C = tm.zeros([[4,4] for _ in range(L)])
    for l in range(0,L+1):
        P = get_level_mapping_matrix_as_tt(L, l)
        PP = P @ P.copy().transpose()
        PP = PP.copy().expand_dims([1,3]) * PP.copy().expand_dims([0,2])
        PP.reshape_mode_indices([4,4])
        C = (C + 2**(-l) * PP).reapprox(ranks_new=64)
    return C

def solve_PDE_1D(f, **solver_options):
    L = len(f) - 1
    g = (get_rhs_matrix_as_tt(L) @ f).squeeze()
    A = get_laplace_matrix_as_tt(L)
    for i in range(len(A)):
        g.tensors[i] /= 2

    u, r2 = solve_with_grad_descent(A, g, **solver_options)
    D = get_derivative_matrix_as_tt(L)
    Du = (D @ u).reapprox(rel_error=1e-15)
    return u, Du, r2

def solve_PDE_2D(f: tm.TensorTrain, **solver_options):
    f = f.copy().flatten_mode_indices()
    L = len(f) - 1
    g = (get_rhs_matrix_as_tt_2D(L) @ f).squeeze()
    A = build_laplace_matrix_2D(L)
    for i in range(len(A)):
        g.tensors[i] /= 4

    u, r2 = solve_with_grad_descent(A, g, **solver_options)
    return u, r2

def solve_PDE_1D_with_preconditioner(f, **solver_options):
    L = len(f) - 1
    g = (get_rhs_matrix_as_tt(L) @ f).squeeze()
    A = get_laplace_matrix_as_tt(L)
    for i in range(len(A)):
        g.tensors[i] /= 2

    C = get_bpx_preconditioner(L)
    B = (C @ A @ C).reapprox(rel_error=1e-12)
    b = (C @ g).reapprox(rel_error=1e-12)

    v, r2 = solve_with_grad_descent(B, b, **solver_options)
    u = (C @ v).reapprox(rel_error=1e-16)

    D = get_derivative_matrix_as_tt(L)
    for i in range(L):
        D.tensors[i] = D.tensors[i] * np.sqrt(2)
    DC = (D @ C).reapprox(rel_error=1e-16)
    Du = (DC @ v).reapprox(rel_error=1e-16)
    return u, Du, r2

def solve_PDE_2D_with_preconditioner(f, max_rank=20, **solver_options):
    print("Building LHS and RHS...")
    L = len(f) - 1
    f = f.copy().flatten_mode_indices()
    g = (get_rhs_matrix_as_tt_2D(L) @ f).squeeze().reapprox(rel_error=1e-12)
    A = build_laplace_matrix_2D(L).reapprox(ranks_new=max_rank)
    for i in range(len(A)):
        g.tensors[i] /= 4

    C = get_bpx_preconditioner_by_sum_2D(L).reapprox(rel_error=1e-12)
    AC = (A @ C).reapprox(ranks_new=max_rank)
    B = (C @ AC).reapprox(ranks_new=max_rank)
    b = (C @ g).reapprox(ranks_new=max_rank)

    print("Starting solver....")
    v, r2 = solve_with_grad_descent(B, b, max_rank=max_rank, **solver_options)
    u = (C @ v).reapprox(ranks_new=max_rank)
    return u, r2


def solve_with_grad_descent(A, b, n_steps_max=200, lr=0.8, max_rank=20, print_steps=False, recalc_residual_every_n=10, r2_accuracy=None):
    x = tm.zeros(b.mode_sizes)
    for n in range(n_steps_max):
        if n%recalc_residual_every_n == 0:
            r = (b - A @ x).reapprox(ranks_new=max_rank)
        Ar = (A @ r).reapprox(ranks_new=max_rank)
        r2 = r.norm_squared()
        if r2_accuracy and (r2 <= r2_accuracy):
            break
        gamma = float(lr * r2 / (r @ Ar).eval().flatten())

        x = x + gamma * r
        r = r - gamma * Ar
        x.reapprox(ranks_new=max_rank)
        r.reapprox(ranks_new=max_rank)
        if print_steps and (n%5 == 0):
            print(f"Step {n:4d}: ||r||² = {r2:.2e} / {r2_accuracy:.2e}, gamma = {gamma: .2e}")
    r = (b - A @ x).reapprox(ranks_new=max_rank)
    r2 = r.norm_squared()
    print(f"Final r2: {r2:.2e}")
    return x, r2

def _get_bpx_factors():
    U = np.zeros([4, 2, 2, 2])
    U[0,0,0,0] = 1
    U[0,1,0,0] = -1
    U[1,0,1,0] = -1
    U[0,1,1,0] = 1
    U[2,1,0,1] = 1
    U[3,0,0,1] = -1
    U[2,1,0,1] = 1
    U[3,0,0,1] = -1
    U[0,0,1,1] = 1
    U[0,1,1,1] = -1
    
    V = np.zeros([4, 2, 2, 4])
    V[:,0,0,0] = 1/4
    V[0,1,0,0] = 1/2
    V[2,1,0,0] = 1/2
    V[0,0,1,0] = 1/2
    V[1,0,1,0] = 1/2
    V[0,1,1,0] = 1
    V[1,0,0,1] = 1/2
    V[3,0,0,1] = 1/2
    V[:,1,0,1] = 1/4
    V[0,1,1,1] = 1/2
    V[1,1,1,1] = 1/2
    V[1,0,1,1] = 1
    V[2,0,0,2] = 1/2
    V[3,0,0,2] = 1/2
    V[2,1,0,2] = 1
    V[0,1,1,2] = 1/2
    V[2,1,1,2] = 1/2
    V[:,0,1,2] = 1/4
    V[2,1,0,3] = 1/2
    V[3,1,0,3] = 1/2
    V[3,0,0,3] = 1
    V[1,0,1,3] = 1/2
    V[3,0,1,3] = 1/2
    V[:,1,1,3] = 1/4
    
    W = np.zeros([4, 2, 2, 4])
    W[0,0,0,0] = 1
    W[0,1,1,0] = 1
    W[0,1,0,1] = 1
    W[1,0,1,1] = 1
    W[2,1,0,2] = 1
    W[0,0,1,2] = 1
    W[3,0,0,3] = 1
    W[0,1,1,3] = 1
    
    Y = np.zeros([2, 2, 2, 2])
    Y[:,:,0,0] = 1/4
    Y[0,:,1,0] = 1/2
    Y[1,:,0,1] = 1/2
    Y[:,:,1,1] = 1/4
    
    return W,U,V,Y

def get_bpx_preconditioner(L):
    W, _, V, _ = _get_bpx_factors()
    W *= 0.5
    V *= 0.5
    X = np.zeros([8,2,2,8])
    X[:4,:,:,:4] = W
    X[:4,:,:,4:] = W
    X[4:,:,:,4:] = V

    C = np.array([1,0,0,0, 1,0,0,0], dtype=float).reshape([1,1,1,8])
    Z = np.array([0,0,0,0, 1,0,0,0], dtype=float).reshape([8,1,1,1])
    return tm.TensorTrain([C] + [X for _ in range(L)] + [Z]).squeeze()


def build_mass_matrix_legendre(L):
    factors = [(np.eye(2)/2)[None, ..., None] for _ in range(L)]
    factors += [_get_gram_matrix_legendre()[None, ..., None] / 2]
    return tm.TensorTrain(factors)

def build_mass_matrix_in_nodal_basis(L):
    M = get_overlap_matrix_as_tt(L)
    M.tensors[-1] = M.tensors[-1][...,None,:]
    M_legendre = build_mass_matrix_legendre(L)
    return (M.copy().transpose() @ M_legendre @ M).squeeze().reapprox()

def build_2D_mass_matrix(L):
    mass = build_mass_matrix_in_nodal_basis(L)
    mx = mass.copy().expand_dims([1, 3])
    my = mass.copy().expand_dims([0, 2])
    mass = mx * my
    mass.reshape_mode_indices([4,4])
    return mass

def build_laplace_matrix_2D(L):
    A = get_laplace_matrix_as_tt(L)
    M = build_mass_matrix_in_nodal_basis(L)
    Ax = A.copy().expand_dims([1, 3])
    Ay = A.copy().expand_dims([0, 2])
    Mx = M.copy().expand_dims([1, 3])
    My = M.copy().expand_dims([0, 2])
    return (Ax * My + Ay * Mx).reshape_mode_indices([4,4]).reapprox(rel_error=1e-15)