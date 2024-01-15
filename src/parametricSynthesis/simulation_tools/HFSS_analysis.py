'''
Goal of this module is to take in an s2p file from HFSS and analyze the results
in the context of a desired filter network.
'''

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import skrf as rf
from scipy.interpolate import interp1d
from scipy.optimize import newton
from ..network_tools.component_ABCD import DegenerateParametricInverter_Amp, ABCD_to_S, ABCD_to_Z
from dataclasses import dataclass
from scipy.optimize import root_scalar

'''
overall pipeline: 
- import an s2p file using skrf
- create an interpolated function for each  ABCD parameter of the network, and is mirrored about the x-axis with odd symmetry, 
  so that f(omega) = -f(-omega)
- convert the s parameters to an ABCD matrix function, which will return an ABCD matrix at all values of omega
- attach an inverter element from component_ABCD.py, and create a function that returns the total ABCD matrix from the 
perspective of the signal and idler ports
- create a function that returns the S matrix from the ABCD matrix as a function of inductance, coupling rate on the inverter, and frequency
'''

def import_s2p(filename):
  '''
  imports an s2p file and returns a skrf network object
  '''
  return rf.Network(filename)

#below function is not needed because the skrf network object handles it already
# def index_to_abcd(i, j, skrf_network):
#     '''
#     converts a pair of indices to the corresponding ABCD parameter in an skrf network object
#     :param i:
#     :param j:
#     :return: correct ABCD method from skrf network object
#     '''
#     if i == 0 and j == 0:
#         return skrf_network.a
#     elif i == 0 and j == 1:
#         return skrf_network.b
#     elif i == 1 and j == 0:
#         return skrf_network.c
#     elif i == 1 and j == 1:
#         return skrf_network.d
#     else:
#         raise ValueError('invalid index')

def sum_real_and_imag(freal, fimag):
    '''
    takes in two functions that return real and imaginary parts of a complex function, and returns a function that returns the
    complex function
    :param freal:
    :param fimag:
    :return:
    '''
    def fcomplex(x):
        return freal(x) + 1j*fimag(x)
    return fcomplex

def interpolate_mirrored_ABCD_functions(skrf_network, omega):
    #first take the skrf_network and extend the frequency range to negative frequencies, conjugating the ABCD matrix at negative frequencies
    #then interpolate each ABCD parameter
    skrf_network.f_mirror = np.insert(skrf_network.f, 0, -np.flip(skrf_network.f))
    # print("DEBUG skrf_net_shapes:", skrf_network.a.shape, np.flip(np.conjugate(skrf_network.a)).shape)
    skrf_network.a_mirror = np.concatenate([np.flip(np.conjugate(skrf_network.a)), skrf_network.a])
    res = sum_real_and_imag(interp1d(skrf_network.f_mirror, skrf_network.a_mirror.real, axis = 0), interp1d(skrf_network.f_mirror, skrf_network.a_mirror.imag, axis = 0))(omega/2/np.pi)
    return res


# def evaluate_net_ABCD_functions(f_arr, interpolated_mirrored_ABCD_functions):
#   '''
#   takes in an array of frequencies and returns a matrix of the evaluated ABCD parameters of the network in a 2x2xN shape
#   '''
#   return np.array([[interpolated_mirrored_ABCD_functions[0,0](f_arr), interpolated_mirrored_ABCD_functions[0,1](f_arr)],
#                    [interpolated_mirrored_ABCD_functions[1,0](f_arr), interpolated_mirrored_ABCD_functions[1,1](f_arr)]])

'''
omega0_val: float
  omega1: sp.Symbol
  omega2: sp.Symbol
  L: sp.Symbol
  R_active: sp.Symbol
  Jpa_sym: sp.Symbol
  dw: float
'''
@dataclass
class interpolated_network_with_inverter_from_filename:
    '''
    takes in an s
    '''
    filename: str
    L_sym: sp.Symbol
    inv_J_sym: sp.Symbol
    omega0_val: float
    dw: float

    def __post_init__(self):
        omega_signal, omega_idler, R_active = sp.symbols("omega, omega_i, R")
        self.skrf_network = import_s2p(self.filename)
        self.inverter = DegenerateParametricInverter_Amp(omega1 = omega_signal,
                                                         omega2 = omega_idler,
                                                         omega0_val = self.omega0_val,
                                                         L = self.L_sym,
                                                         R_active = R_active,
                                                         Jpa_sym = self.inv_J_sym
                                                         )
        self.signal_inductor = self.inverter.signal_inductor
        self.signal_inductor_ABCD_func_ = sp.lambdify([self.signal_inductor.omega_symbol, self.signal_inductor.symbol],
                                                      self.signal_inductor.ABCDshunt())
        self.inverter_ABCD = self.inverter.ABCD_shunt().subs(omega_idler, omega_signal-2*self.omega0_val)#this makes it totally degenerate

    def ind_ABCD_mtx_func_(self, L, omega):
        return
    def ind_ABCD_mtx_func(self, L_arr, omega_arr):
        # becuse sympy's lambdify is absolutely *shit* at broadcasting, we need to do something like this
        res_mtx_arr = []
        for L, omega in zip(L_arr, omega_arr):
            res_mtx = self.signal_inductor_ABCD_func_(L, omega)
            res_mtx_arr.append(res_mtx)
        res_mtx_arr_np = np.array(res_mtx_arr, dtype='complex')
        return res_mtx_arr_np

    def evaluate_ABCD_mtx(self, L_arr, Jpa_arr, omega_arr, active = True):
        '''
        returns the S matrix of the network for each inductance, inverter coupling rate, and frequency in the input arrays
        This can only handle variation over frequency! The inductance and coupling rate must be constant but match the length of the frequency array
        '''
        self.inv_ABCD_mtx_func = sp.lambdify([self.inverter.L, self.inverter.Jpa_sym, self.inverter.omega1],
                                             self.inverter_ABCD)

        # this will return a 2x2xN matrix of floats, with 1xN input arrays

        self.s2p_net_ABCD_mtx_signal = interpolate_mirrored_ABCD_functions(self.skrf_network, omega_arr)
        self.s2p_net_ABCD_mtx_idler = interpolate_mirrored_ABCD_functions(self.skrf_network,
                                                                          omega_arr - 2 * self.omega0_val)

        # these will also return a Nx2x2 matrix of floats, with 1xN input

        # np.matmul needs Nx2x2 inputs to treat the last two as matrices,
        # so we use moveaxis to rearrange the first and last axis so that
        # [
        # [[1,2,3],[4,5,6]]
        # [[7,8,9],[10,11,12]]
        # ]
        # becomes
        # [
        # [[1,4],
        #  [7,10]],
        # [[2,5],
        #  [8,11]],
        # [[3, 6],
        #  [9,12]],
        # ]
        def mv(arr):
            return np.moveaxis(arr, -1, 0)
        if active:
            total_ABCD_mtx_evaluated = np.matmul(
                np.matmul(
                    self.s2p_net_ABCD_mtx_signal,
                    mv(self.inv_ABCD_mtx_func(L_arr, Jpa_arr, omega_arr))
                ), self.s2p_net_ABCD_mtx_idler)
        else:
            signal_inductor_ABCD_array = self.ind_ABCD_mtx_func(omega_arr, L_arr)
            # print("Debug: signal inductor ABCD array shape:", signal_inductor_ABCD_array.shape)
            total_ABCD_mtx_evaluated = np.matmul(
                self.s2p_net_ABCD_mtx_signal,
                self.ind_ABCD_mtx_func(L_arr, omega_arr)
            )

        # now we have a total Nx2x2 ABCD matrix, but to convert that to a scattering matrix, we need the 2x2xN shape back
        # so we use moveaxis again
        self.total_ABCD_mtx_evaluated_reshaped = np.moveaxis(total_ABCD_mtx_evaluated, 0, -1)

        return self.total_ABCD_mtx_evaluated_reshaped

        # now we can convert to S parameters using the helper function I already have
    def evaluate_Smtx(self, L_arr, Jpa_arr, omega_arr):

        return ABCD_to_S(self.evaluate_ABCD_mtx(L_arr, Jpa_arr, omega_arr), 50, num = True)

    def compare_ABCD_to_ideal(self, analytical_net):
        '''
        compares the ABCD matrix of the simulated network to the ideal ABCD matrix of the filter circuit as computed from the network_synthesis module
        :param analytical_net:
        :return:
        '''
        pass

    def find_p2_input_impedance(self, L_arr, omega_arr, Z0 = 50):
        '''
        returns the impedance seen from the inverter
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_ABCD_mtx(L_arr, np.array([0]), omega_arr, active = False)
        self.filterZmtxs = ABCD_to_Z(ABCD_mtxs, num=True)
        Z = self.filterZmtxs
        #to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        #impedance matrix and the source impedance
        port2_input_impedance = Z[1,1]-Z[1,0]*Z[0,1]/(Z[0,0]+Z0)

        return port2_input_impedance

    def find_p1_input_impedance(self, L_arr, omega_arr, Zl = 50):
        '''
        returns the impedance seen from the environment
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_ABCD_mtx(L_arr, [0], omega_arr, active = False)
        self.filterZmtxs = ABCD_to_Z(ABCD_mtxs,  num=True)
        Z = self.filterZmtxs
        #to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        #impedance matrix and the source impedance
        port1_input_impedance = Z[0,0]-Z[1,0]*Z[0,1]/(Z[1,1]+Zl)

        return port1_input_impedance

    def find_modes_from_input_impedance(self, Lval, omega_arr, Z0 = 50, debug = False):
        '''
        returns the modes as a function of the inductance of the array inductor, as well as
        the real part of the impedance at the root and the slope of the imaginary part at the root
        '''

        self.p2_input_impedance = self.find_p2_input_impedance(Lval*np.ones_like(omega_arr), omega_arr, Z0 = Z0)
        #for each inductance, we need to find a list of zeroes of the imaginary part of the admittance in some frequency range
        self.p2_input_admittance = 1/self.p2_input_impedance
        omega_step = omega_arr[1]-omega_arr[0]
        #now build an interpolation function for both real and the imaginary part of the admittance
        re_f = interp1d(omega_arr, np.real(self.p2_input_admittance), kind = 'cubic')
        im_f = interp1d(omega_arr, np.imag(self.p2_input_admittance), kind = 'cubic')
        im_fp = interp1d(omega_arr, np.imag(np.gradient(self.p2_input_admittance)/omega_step), kind = 'cubic')

        #we have to find our initial guesses, which I will get from the number of flips of the sign of the imaginary part of the admittance
        #this is a bit of a hack, but it should work
        #first, find the sign of the imaginary part of the admittance
        sign = np.sign(np.imag(self.p2_input_admittance))
        #now find the number of sign flips
        sign_flips = np.diff(sign)
        #now find the indices of the sign flips
        sign_flip_indices = np.where(sign_flips != 0)[0]
        #now find the frequencies at which the sign flips
        sign_flip_freqs = omega_arr[sign_flip_indices]
        roots = np.empty(sign_flip_freqs.size)
        reY_at_roots = np.empty(sign_flip_freqs.size)
        imYp_at_roots = np.empty(sign_flip_freqs.size)
        for i, flip_freq in enumerate(sign_flip_freqs):
            if debug: print("Debug: sign flip at", flip_freq/2/np.pi/1e9, " GHz")
            root = newton(im_f, flip_freq, maxiter = 1000)
            roots[i] = root
            if debug: print('Debug: Root at ',i, root/2/np.pi/1e9, " GHz")
            reY_at_roots[i] = re_f(root)
            imYp_at_roots[i] = im_fp(root)

        return roots, reY_at_roots, imYp_at_roots

    def modes_as_function_of_inductance(self, L_arr, omega_arr, Z0=50):
        roots, reY_at_roots, imYp_at_roots = [], [], []
        for Lval in L_arr:
            print("Inductance value: ", Lval*1e12, " pH")
            res = self.find_modes_from_input_impedance(Lval, omega_arr, Z0 = Z0)
            roots.append(res[0])
            reY_at_roots.append(res[1])
            imYp_at_roots.append(res[2])
        return np.array(roots), np.array(reY_at_roots), np.array(imYp_at_roots)
