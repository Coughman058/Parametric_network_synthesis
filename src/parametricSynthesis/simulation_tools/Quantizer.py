from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import newton

'''
The need for quantization extends to both the analytical circuit and the simulation,
so the goal for this file is to build a black-box quantizer that can work for both

ideally it would input an interpolation function for the input admittance seen from the inverter, then return the modes
and Zpeff, with effective inductances as well
'''


def sum_real_and_imag(freal, fimag):
    '''
    takes in two functions that return real and imaginary parts of a complex function, and returns a function that returns the
    complex function
    :param freal:
    :param fimag:
    :return:
    '''

    def f_complex(x):
        return freal(x) + 1j * fimag(x)

    return f_complex


def find_modes_from_input_impedance(p2_input_impedance, omega_arr, debug=False):
    '''
    returns the modes as a function of the inductance of the array inductor, as well as
    the real part of the impedance at the root and the slope of the imaginary part at the root
    '''
    # a check to make sure the input array dtype is complex
    assert p2_input_impedance.dtype == np.complex128, "input impedance must be complex"

    # self.find_p2_input_impedance(Lval*np.ones_like(omega_arr), omega_arr, Z0 = Z0)
    # for each inductance, we need to find a list of zeroes of the imaginary part of the admittance in some frequency range
    p2_input_admittance = 1 / p2_input_impedance
    omega_step = omega_arr[1] - omega_arr[0]
    # now build an interpolation function for both real and the imaginary part of the admittance
    re_f = interp1d(omega_arr, np.real(p2_input_admittance), kind='cubic')
    im_f = interp1d(omega_arr, np.imag(p2_input_admittance), kind='cubic')
    im_fp = interp1d(omega_arr, np.imag(np.gradient(p2_input_admittance) / omega_step), kind='cubic')
    im_fpp = interp1d(omega_arr, np.imag(np.gradient(np.gradient(p2_input_admittance)) / omega_step**2), kind='cubic')
    # we have to find our initial guesses, which I will get from the number of flips of the sign of the imaginary part of the admittance
    # this is a bit of a hack, but it should work
    # first, find the sign of the imaginary part of the admittance
    sign = np.sign(np.imag(p2_input_admittance))
    # now find the number of sign flips
    sign_flips = np.diff(sign)
    # now find the indices of the sign flips
    sign_flip_indices = np.where(sign_flips != 0)[0]
    # now find the frequencies at which the sign flips
    sign_flip_freqs = omega_arr[sign_flip_indices]
    roots = np.empty(sign_flip_freqs.size)
    reY_at_roots = np.empty(sign_flip_freqs.size)
    imYp_at_roots = np.empty(sign_flip_freqs.size)
    imYpp_at_roots = np.empty(sign_flip_freqs.size)
    for i, flip_freq in enumerate(sign_flip_freqs):
        if debug: print("Debug: sign flip at", flip_freq / 2 / np.pi / 1e9, " GHz")
        root = newton(im_f, flip_freq, maxiter=1000)
        roots[i] = root
        if debug: print('Debug: Root at ', i, root / 2 / np.pi / 1e9, " GHz")
        reY_at_roots[i] = re_f(root)
        imYp_at_roots[i] = im_fp(root)
        imYpp_at_roots[i] = im_fpp(root)

    return roots, reY_at_roots, imYp_at_roots, imYpp_at_roots

def mode_results_to_device_params(res):
    '''
    takes the results of the find_modes_from_input_impedance function and returns the device parameters in terms of
    quality factor Q,
    capacitance C,
    inductance L,
    and Zpeff
    '''

    # unpack the results
    roots, reY_at_roots, imYp_at_roots, imYpp_at_roots = res
    # now find the quality factor
    q = np.abs(roots) /2 * (imYp_at_roots/reY_at_roots)
    c = 1 / 2 * imYp_at_roots
    Zpeff = 1/roots/c
    L = imYpp_at_roots**2/4/c**3

    return q, c, L, Zpeff
