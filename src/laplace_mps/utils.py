from typing import List
import numpy
import numpy as np
import numpy.polynomial.polynomial
import laplace_mps.tensormethods as tm

REL_ERROR = 1e-15

def eval_poly(x, coeffs):
    y = np.zeros_like(x)
    for k, c in enumerate(coeffs):
        y += c * x**k
    return y

def draw_vertical_grid(L, ax=None):
    L = min(L, 6)
    ax = ax or plt.gca()
    for i in range(2**L + 1):
        ax.axvline(i*2**(-L), color='k', alpha=0.1)


class TrigFunction:
    def __init__(self, a,b, omega):
        self.a  = a
        self.b = b
        self.omega = omega

    def eval(self, x):
        return self.a * np.cos(self.omega * x) + self.b * np.sin(self.omega * x)

    def derive(self):
        return TrigFunction(self.b * self.omega, -self.a * self.omega, self.omega)

    @property
    def coeffs(self):
        return self.a, self.b, self.omega / (2 * np.pi)


if __name__ == '__main__':
    poly_coefs = [1,2,3]
    p = numpy.polynomial.polynomial.Polynomial(poly_coefs)
    p_leg = p.convert(domain=[0,1], kind=numpy.polynomial.legendre.Legendre)
    print(p)
    print(p_leg)


def build_u_with_correct_boundary_conditions(poly_coeffs, trig_coeffs):
    trig_functions = [TrigFunction(c[0], c[1], c[2] * 2 * np.pi) for c in trig_coeffs]
    trig_right = np.sum([t.eval(1) for t in trig_functions])
    trig_prime_right = np.sum([t.derive().eval(1) for t in trig_functions])
    ratio = trig_prime_right / trig_right

    poly_coeffs[0] = 0
    coeff_sum = np.sum([(ratio + i) * poly_coeffs[i] for i in range(2, len(poly_coeffs))])
    poly_coeffs[1] = -coeff_sum / (1 + ratio)
    poly = np.polynomial.Polynomial(poly_coeffs)
    return poly, trig_functions


def get_f_from_u(poly: np.polynomial.Polynomial, trig: List[TrigFunction], L):
    trig0_tt = tm.zeros([(2,) for _ in range(L+1)])
    trig1_tt = tm.zeros([(2,) for _ in range(L+1)])
    trig2_tt = tm.zeros([(2,) for _ in range(L+1)])
    for t in trig:
        trig0_tt = trig0_tt + get_trig_function_as_tt(t.coeffs, L)
        trig1_tt = trig1_tt + get_trig_function_as_tt(t.derive().coeffs, L)
        trig2_tt = trig2_tt + get_trig_function_as_tt(t.derive().derive().coeffs, L)

    poly0_tt = get_polynomial_as_tt(poly.coef, L)
    poly1_tt = get_polynomial_as_tt(poly.deriv(1).coef, L)
    poly2_tt = get_polynomial_as_tt(poly.deriv(2).coef, L)

    f = trig0_tt * poly2_tt + 2 * trig1_tt * poly1_tt + trig2_tt * poly0_tt
    return -f.reapprox(rel_error=1e-16)


def evaluate_f_from_u(poly: np.polynomial.Polynomial, trig: List[TrigFunction], x):
    trig0_eval = np.zeros_like(x)
    trig1_eval = np.zeros_like(x)
    trig2_eval = np.zeros_like(x)
    for t in trig:
        trig0_eval += t.eval(x)
        trig1_eval += t.derive().eval(x)
        trig2_eval += t.derive().derive().eval(x)

    poly0_eval = poly(x)
    poly1_eval = poly.deriv(1)(x)
    poly2_eval = poly.deriv(2)(x)
    f = poly0_eval * trig2_eval + 2 * poly1_eval * trig1_eval + poly2_eval * trig0_eval
    return -f


def eval_function(poly, trig_functions, x):
    trig = np.zeros_like(x)
    for t in trig_functions:
        trig += t.eval(x)
    return trig * poly(x)


def get_u_function_as_tt(poly, trig: List[TrigFunction], L):
    u_tt = get_trig_function_as_tt(trig[0].coeffs, L)
    for t in trig[1:]:
        u_tt = u_tt + get_trig_function_as_tt(t.coeffs, L)
    u_tt = u_tt * get_polynomial_as_tt(poly.coef, L)
    return u_tt.reapprox(rel_error=1e-16)


def get_example_u_1D(L, basis='corner'):
    u = get_polynomial_as_tt([1, -3, 2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    if basis == 'corner':
        s = np.array([-1, 1])
    elif basis == 'nodal':
        s = np.array([1])
    else:
        raise ValueError(f"Unknown basis: {basis}")
    return evaluate_nodal_basis(u, s).squeeze().reapprox(rel_error=REL_ERROR)

def get_example_u_deriv_1D(L, basis='corner'):
    u1 = get_polynomial_as_tt([-3, 4], L) * get_trig_function_as_tt([0, 1, 1.0], L) + \
        get_polynomial_as_tt([2*np.pi, -6*np.pi, 4*np.pi], L) * get_trig_function_as_tt([1, 0, 1.0], L)
    if basis == 'corner':
        s = np.array([-1, 1])
    elif basis == 'nodal':
        s = np.array([1])
    else:
        raise ValueError(f"Unknown basis: {basis}")
    return evaluate_nodal_basis(u1, s).squeeze().reapprox(rel_error=REL_ERROR)

def get_example_f_1D(L):
    f = get_polynomial_as_tt([4-4*np.pi**2, 12*np.pi**2, -8*np.pi**2], L) * get_trig_function_as_tt([0, 1, 1.0], L) + \
          get_polynomial_as_tt([-12*np.pi, 16*np.pi], L) * get_trig_function_as_tt([1, 0, 1.0], L)
    return -f.reapprox(rel_error=REL_ERROR)


def get_example_u_2D(L, basis='corner'):
    ux = get_polynomial_as_tt([0, -1, 5, -3], L)
    uy = get_polynomial_as_tt([1, -3, 2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    u = ux.expand_dims(1) * uy.expand_dims(0)
    if basis == 'corner':
        s = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]])
    elif basis == 'nodal':
        s = np.ones([2,1])
    else:
        raise ValueError(f"Unknown basis: {basis}")
    return evaluate_nodal_basis(u, s).squeeze()

def get_example_f_2D(L):
    ux = get_polynomial_as_tt([0, -1, 5, -3], L)
    uy = get_polynomial_as_tt([1, -3, 2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    ux2 = get_polynomial_as_tt([10, -18], L)
    uy2 = get_polynomial_as_tt([4-4*np.pi**2, 12*np.pi**2, -8*np.pi**2], L) * get_trig_function_as_tt([0, 1, 1.0], L)
    uy2 = uy2 + get_polynomial_as_tt([-12*np.pi, 16*np.pi], L) * get_trig_function_as_tt([1, 0, 1.0], L)

    f = ux.expand_dims(1) * uy2.expand_dims(0) + ux2.expand_dims(1) * uy.expand_dims(0)
    return -f.reapprox(rel_error=REL_ERROR)

def kronecker_prod_2D(A, B):
    C = A.copy().expand_dims([1,3]) * B.copy().expand_dims([0,2])
    new_shapes = [(U.shape[1]*U.shape[2], U.shape[3]*U.shape[4]) for U in C]
    return C.reshape_mode_indices(new_shapes)


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