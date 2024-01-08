'''
Goal of this module is to take in an s2p file from HFSS and analyze the results
in the context of a desired filter network.
'''

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import skrf as rf
from scipy.interpolate import interp1d
from ..network_tools.component_ABCD import DegenerateParametricInverter_Amp, ABCD_to_S
from dataclasses import dataclass

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
  skrf_network.f = np.insert(skrf_network.f, 0, -np.flip(skrf_network.f))
  skrf_network.a = np.concatenate(np.flip(np.conjugate(skrf_network.a), axis = 0), skrf_network.a)
  res = sum_real_and_imag(interp1d(skrf_network.f, skrf_network.a.real, axis = 0), interp1d(skrf_network.f, skrf_network.a.imag, axis = 0))(omega/2/np.pi)
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
        self.inverter_ABCD = self.inverter.ABCD_shunt().subs(omega_idler, omega_signal-2*self.omega0_val)#this makes it totally degenerate

    def evaluate_Smtx(self, L_arr, Jpa_arr, omega_arr):
        '''
        returns the S matrix of the network for each inductance, inverter coupling rate, and frequency in the input arrays
        '''
        self.inv_ABCD_mtx_func = sp.lambdify([self.inverter.L, self.inverter.Jpa_sym, self.inverter.omega1], self.inverter_ABCD)
        #this will return a 2x2xN matrix of floats, with 1xN input arrays

        self.s2p_net_ABCD_mtx_signal = interpolate_mirrored_ABCD_functions(self.skrf_network, omega_arr)
        self.s2p_net_ABCD_mtx_idler = interpolate_mirrored_ABCD_functions(self.skrf_network, omega_arr-2*self.omega0_val)
        #these will also return a 2x2xN matrix of floats, with 1xN input

        #np.matmul needs Nx2x2 inputs to treat the last two as matrices,
        #so we use moveaxis to rearrange the first and last axis so that
        # [
        # [[1,2,3],[4,5,6]]
        # [[7,8,9],[10,11,12]]
        # ]
        #becomes
        # [
        # [[1,4],
        #  [7,10]],
        # [[2,5],
        #  [8,11]],
        # [[3, 6],
        #  [9,12]],
        # ]
        def mv(arr):
            return np.moveaxis(arr, -1,0)

        total_ABCD_mtx_evaluated = np.matmul(
            np.matmul(
                mv(self.s2p_net_ABCD_mtx_signal),
                mv(self.inv_ABCD_mtx_func(L_arr, Jpa_arr, omega_arr))
            ), mv(self.s2p_net_ABCD_mtx_idler))

        #now we have a total Nx2x2 ABCD matrix, but to convert that to a scattering matrix, we need the 2x2xN shape back
        #so we use moveaxis again
        self.total_ABCD_mtx_evaluated_reshaped = np.moveaxis(total_ABCD_mtx_evaluated, 0, -1)

        #now we can convert to S parameters using the helper function I already have
        return ABCD_to_S(self.total_ABCD_mtx_evaluated_reshaped, 50, num = True)


