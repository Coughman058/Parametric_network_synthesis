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
from ..network_tools.component_ABCD import DegenerateParametricInverterAmp, abcd_to_s, abcd_to_z
from dataclasses import dataclass
from scipy.optimize import root_scalar
from ..simulation_tools.Quantizer import sum_real_and_imag, find_modes_from_input_impedance, mode_results_to_device_params
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


def interpolate_mirrored_abcd_functions(skrf_network, omega):
    #first take the skrf_network and extend the frequency range to negative frequencies, conjugating the ABCD matrix at negative frequencies
    #then interpolate each ABCD parameter
    skrf_network.f_mirror = np.insert(skrf_network.f, 0, -np.flip(skrf_network.f))
    # print("DEBUG skrf_net_shapes:", skrf_network.a.shape, np.flip(np.conjugate(skrf_network.a)).shape)
    skrf_network.a_mirror = np.concatenate([np.flip(np.conjugate(skrf_network.a)), skrf_network.a])
    res = sum_real_and_imag(interp1d(skrf_network.f_mirror, skrf_network.a_mirror.real, axis = 0), interp1d(skrf_network.f_mirror, skrf_network.a_mirror.imag, axis = 0))(omega/2/np.pi)
    return res


@dataclass
class InterpolatedNetworkWithInverterFromFilename:
    '''
    This file takes in an s2p file from HFSS and analyzes the results by finding modes and computing nonlinearities
    '''
    filename: str
    L_sym: sp.Symbol
    inv_J_sym: sp.Symbol
    omega0_val: float
    dw: float

    def __post_init__(self):
        omega_signal, omega_idler, r_active = sp.symbols("omega, omega_i, R")
        self.skrf_network = import_s2p(self.filename)
        self.inverter = DegenerateParametricInverterAmp(omega1 = omega_signal,
                                                        omega2 = omega_idler,
                                                        omega0_val = self.omega0_val,
                                                        L = self.L_sym,
                                                        R_active = r_active,
                                                        Jpa_sym = self.inv_J_sym
                                                        )
        self.signal_inductor = self.inverter.signal_inductor
        self.signal_inductor_ABCD_func_ = sp.lambdify([self.signal_inductor.omega_symbol, self.signal_inductor.symbol],
                                                      self.signal_inductor.ABCDshunt())
        inverter_total_abcd = self.inverter.abcd_signal_inductor_shunt() * self.inverter.inverter_shunt() * self.inverter.abcd_idler_inductor_shunt()
        self.inverter_ABCD = inverter_total_abcd.subs(omega_idler, omega_signal-2*self.omega0_val)#this makes it totally degenerate

    def ind_abcd_mtx_func(self, L_arr, omega_arr):
        # becuse sympy's lambdify is absolutely *shit* at broadcasting, we need to do something like this
        res_mtx_arr = []
        for L, omega in zip(L_arr, omega_arr):
            res_mtx = self.signal_inductor_ABCD_func_(L, omega)
            res_mtx_arr.append(res_mtx)
        res_mtx_arr_np = np.array(res_mtx_arr, dtype='complex')
        return res_mtx_arr_np

    def evaluate_abcd_mtx(self, L_val, Jpa_val, omega_arr, active = True):
        '''
        returns the ABCD matrix of the network for each inductance, inverter coupling rate, and frequency in the input arrays
        This can only handle variation over frequency! The inductance and coupling rate must be a single constant
        '''
        if np.size(L_val) > 1 or np.size(Jpa_val) > 1:
            raise Exception("Improper variation in ABCD mtx eval, use only one inductance at a time")
        L_arr = L_val*np.ones_like(omega_arr)
        Jpa_arr = Jpa_val*np.ones_like(omega_arr)
        self.inv_ABCD_mtx_func = sp.lambdify([self.inverter.L, self.inverter.Jpa_sym, self.inverter.omega1],
                                             self.inverter_ABCD)

        # this will return a 2x2xN matrix of floats, with 1xN input arrays

        self.s2p_net_ABCD_mtx_signal = interpolate_mirrored_abcd_functions(self.skrf_network, omega_arr)
        self.s2p_net_ABCD_mtx_idler = interpolate_mirrored_abcd_functions(self.skrf_network,
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
            signal_inductor_ABCD_array = self.ind_abcd_mtx_func(omega_arr, L_arr)
            # print("Debug: signal inductor ABCD array shape:", signal_inductor_ABCD_array.shape)
            total_ABCD_mtx_evaluated = np.matmul(
                self.s2p_net_ABCD_mtx_signal,
                self.ind_abcd_mtx_func(L_arr, omega_arr)
            )

        # now we have a total Nx2x2 ABCD matrix, but to convert that to a scattering matrix, we need the 2x2xN shape back
        # so we use moveaxis again
        self.total_ABCD_mtx_evaluated_reshaped = np.moveaxis(total_ABCD_mtx_evaluated, 0, -1)

        return self.total_ABCD_mtx_evaluated_reshaped

        # now we can convert to S parameters using the helper function I already have
    def evaluate_smtx(self, L_val, Jpa_val, omega_arr):

        return abcd_to_s(self.evaluate_abcd_mtx(L_val, Jpa_val, omega_arr), 50, num = True)

    def compare_abcd_to_ideal(self, analytical_net):
        '''
        compares the ABCD matrix of the simulated network to the ideal ABCD matrix of the filter circuit as computed from the network_synthesis module
        :param analytical_net:
        :return:
        '''
        pass

    def find_p2_input_impedance(self, L_val, omega_arr, Z0 = 50):
        '''
        returns the impedance seen from the inverter
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_abcd_mtx(L_val, 0, omega_arr, active = False)
        self.filterZmtxs = abcd_to_z(ABCD_mtxs, num=True)
        Z = self.filterZmtxs
        #to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        #impedance matrix and the source impedance
        port2_input_impedance = Z[1,1]-Z[1,0]*Z[0,1]/(Z[0,0]+Z0)

        return port2_input_impedance

    def find_p1_input_impedance(self, L_val, omega_arr, Zl = 50):
        '''
        returns the impedance seen from the environment
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_abcd_mtx(L_val, 0, omega_arr, active = False)
        self.filterZmtxs = abcd_to_z(ABCD_mtxs, num=True)
        Z = self.filterZmtxs
        #to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        #impedance matrix and the source impedance
        port1_input_impedance = Z[0,0]-Z[1,0]*Z[0,1]/(Z[1,1]+Zl)

        return port1_input_impedance

    def modes_as_function_of_inductance(self, l_arr, omega_arr, debug=False):
        '''
        Takes in an array of inductance values and frequencies
        returns the modes as a function of the inductance of the array inductor. In the format
        of the return of find_modes_from_input_impedance
        '''

        res_list = []
        res_params_list = []
        for l_val in l_arr:
            if debug: print("Inductance value: ", l_val * 1e12, " pH")
            z_arr = self.find_p2_input_impedance(l_val, omega_arr, Z0=50)
            res = find_modes_from_input_impedance(z_arr, omega_arr, debug=debug)
            res_params = mode_results_to_device_params(res)
            res_list.append(res)
            res_params_list.append(res_params)
        return res_list, res_params_list




