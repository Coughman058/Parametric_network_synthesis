'''
Goal of this module is to calculate protoype coefficients for a negative resistance network for any gain, and using any
prototype polynomial. Butterworth and Chebyshev are the most common, but others are possible.
'''

import numpy as np
from numpy.polynomial import Polynomial as P, Chebyshev as C, Legendre as L, Hermite as H, Laguerre as La
from scipy.special import factorial as fac
def θn_coeff(n, k):
    y = fac(2*n-k)/(fac(n-k)*fac(k)*2**(n-k))
    return y

def bessel_coeffs_from_order(n):
    coeffs = np.zeros(n+1)
    for i in range(n+1):
        coeffs[i] = θn_coeff(n, i)
    return coeffs
def PIL_from_gain(g_db, type ='chebyshev', n = 3, r_db = 0.5):
    """
    calculates the power insertion loss function from the gain of the amplifier on-resonance,
    returns in the form of a polynomial object that we can use for division later
    G: reflection gain of the amplifier in dB on resonance (can set to negative values for passive filters)
    Ripple is specified in dB, defaults to 0.5dB
    """

    #the polynomials include the zeroth order, so we need to add 1 to the order
    g = 10 ** (g_db / 10)
    coeffs = np.zeros(n + 1)
    coeffs[-1] = 1

    if type == 'butterworth':
        g_pl = 4 * g - 2
        A = g_pl / (g_pl - 1)
        k2 = 1 / (g_pl - 2)
        pil_P = A*(1+k2*P(coeffs)**2)

    if type == 'bessel':
        coeffs_P = bessel_coeffs_from_order(n)
        g_pl = 4 * g - 2
        A = g_pl / (g_pl - 1)
        # k2 = 1 / (g_pl - 2)
        bessel_poly = P(coeffs_P)
        pil_P = A * (bessel_poly/bessel_poly(0))**2
        print("PIL at 0: ", pil_P(0))
        print("bessel coeffs: ", coeffs_P)
        print("bessel poly: ", bessel_poly)


    if type == 'chebyshev':
        if n%2 == 0:
            gmin = 10 ** (g_db / 10)
            gmax = 10 ** ((g_db + r_db) / 10)
        else:
            gmin = 10 ** ((g_db - r_db) / 10)
            gmax = 10 ** (g_db / 10)

        gmin_pl = 4*gmin-2
        gmax_pl = 4*gmax-2

        A = gmax_pl/(gmax_pl-1)
        k2 = 1/gmin_pl-1/gmax_pl
        pil_C = A*(1+k2*C(coeffs)**2)
        pil_P = pil_C.convert(kind = P)

    if type == 'legendre':
        Pn = L(coeffs)
        # Pn02 = Pn(0) ** 2
        g_m = 10 ** ((g_db - r_db) / 10)
        g_m_inv = 1 / g_m
        g_pl_inv = 1 / (4 * g - 2)
        gm_pl_inv = 1 / (4 * g_m - 2)
        k2 = (g_pl_inv - gm_pl_inv) / (Pn(0) ** 2 * (1 - g_pl_inv) + gm_pl_inv - 1)
        A = 1 / ((1 + k2) * (1 - gm_pl_inv))
        pil_L = A * (1 + k2 * Pn ** 2)
        pil_P = pil_L.convert(kind=P)

    # now need to multiply times (i)^n to switch variables to j*omega
    coef_new = []
    for i, coef in enumerate(pil_P.coef):
        coef_new.append(coef * (1j) ** i)
    pil = P(coef_new)
    return pil

def poly_from_pil(pil):
    """
    calculates the numerator and denominator to be synthesized in a cauer network
    """

    round_precision = 10
    all_zeros_R = (pil - 1).roots()
    zeros_R = [np.round(root, round_precision) for root in (pil - 1).roots() if
               np.round(root, round_precision).real <= 0]

    all_zeros_D = pil.roots()
    poles_D = [np.round(root, round_precision) for root in pil.roots() if np.round(root.real, round_precision) <= 0]

    num_zeros = len(zeros_R)
    num_poles = len(poles_D)
    # construct R and d from power series polynomials
    R = P.fromroots(zeros_R)
    D = P.fromroots(poles_D)

    z_factor_num = D + R
    z_factor_den = D - R
    num_deg = z_factor_num.degree()
    den_deg = z_factor_den.degree()
    # return z_factor_num, z_factor_den
    return z_factor_num, z_factor_den

# def poly_bessel(g_db, order = 3, debug = False):
#     """
#     calculates the numerator and denominator to be synthesized in a cauer network, specfic to (reverse) bessel filters
#     """
#     g = 10 ** (g_db / 10)
#
#     coeffs_P = bessel_coeffs_from_order(n)
#     g_pl = 4 * g - 2
#     A = g_pl / (g_pl - 1)
#     k2 = 1 / (g_pl - 2)
#     # pil_P = A * (1 + k2 * P(coeffs_P) ** 2)
#     theta_n = P(coeffs_P)
#     if debug: print("bessel coeffs: ", coeffs_P)
#
#     num =
#     den =
#     round_precision = 10
#     all_zeros_R = num.roots()
#     zeros_R = [np.round(root, round_precision) for root in num.roots() if
#                np.round(root, round_precision).real <= 0]
#
#     all_zeros_D = den.roots()
#     poles_D = [np.round(root, round_precision) for root in den.roots() if np.round(root.real, round_precision) <= 0]
#
#     num_zeros = len(zeros_R)
#     num_poles = len(poles_D)
#     # construct R and d from power series polynomials
#     R = P.fromroots(zeros_R)
#     D = P.fromroots(poles_D)
#
#     z_factor_num = D + R
#     z_factor_den = D - R
#     num_deg = z_factor_num.degree()
#     den_deg = z_factor_den.degree()
#     # return z_factor_num, z_factor_den
#     return z_factor_num, z_factor_den

def gs_from_poly(num, den, n, debug = False):
    '''
    calculates the g coefficients from the numerator and denominator polynomials
    '''
    gs = [1]
    for order in range(n):
        quo, rem = divmod(num, den)
        num = den
        den = rem
        gs.append(quo.coef[-1].real)
    gs.append(1/quo.coef[0].real)
    if debug: print('Calculated gs: ', gs)
    return gs

def prototype_gs(G_db, type = 'chebyshev', n = 3, r_db = 0.5, debug = False):
    pil = PIL_from_gain(G_db, type = type, n = n, r_db = r_db)
    num, den = poly_from_pil(pil)
    gs = gs_from_poly(num, den, n)
    return gs

if __name__ == '__main__':
    G_db = 20
    r_db = 0.5
    n = 3

    cheby_gs_active = prototype_gs(G_db, type = 'chebyshev', n = n, r_db = r_db)
    # cheby_gs_passive = prototype_gs(-G_db, type = 'chebyshev', n = n, r_db = r_db)

    # breakpoint()

