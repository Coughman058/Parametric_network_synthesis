'''
Goal of this module is to take in an s2p file from HFSS and analyze the results
in the context of a desired filter network.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import skrf as rf
from scipy.interpolate import interp1d
from scipy.optimize import newton
from ..network_tools.component_ABCD import DegenerateParametricInverterAmp, abcd_to_s, abcd_to_z
from dataclasses import dataclass
from scipy.optimize import root_scalar
from ..simulation_tools.Quantizer import sum_real_and_imag, find_modes_from_input_impedance, mode_results_to_device_params
from ..network_tools.network_synthesis import Network
from tqdm import tqdm
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

class InterpolatedNetworkWithInverterFromSKRF:
    '''
    This file takes in an skrf network, attached an inductive inverter to it, then analyzes the results
    '''

    def __init__(self,
                 skrf_network: rf.Network,
                 L_sym: sp.Symbol,
                 inv_J_sym: sp.Symbol,
                 omega0_val: float,
                 dw: float):
        self.skrf_network = skrf_network
        self.L_sym = L_sym
        self.inv_J_sym = inv_J_sym
        self.omega0_val = omega0_val
        self.dw = dw

        omega_signal, omega_idler, r_active = sp.symbols("omega, omega_i, R")
        self.inverter = DegenerateParametricInverterAmp(omega1=omega_signal,
                                                        omega2=omega_idler,
                                                        omega0_val=self.omega0_val,
                                                        L=self.L_sym,
                                                        R_active=r_active,
                                                        Jpa_sym=self.inv_J_sym
                                                        )
        self.signal_inductor = self.inverter.signal_inductor
        self.signal_inductor_ABCD_func_ = sp.lambdify([self.signal_inductor.omega_symbol, self.signal_inductor.symbol],
                                                      self.signal_inductor.ABCDshunt())
        inverter_total_abcd = self.inverter.abcd_signal_inductor_shunt() * self.inverter.abcd_inverter_shunt() * self.inverter.abcd_idler_inductor_shunt()
        self.inverter_ABCD = inverter_total_abcd.subs(omega_idler,
                                                      omega_signal - 2 * self.omega0_val)  # this makes it totally degenerate

    def ind_abcd_mtx_func(self, L_arr, omega_arr):
        #because sympy's lambdify is absolutely *shit* at broadcasting, we need to do something like this
        res_mtx_arr = []
        for L, omega in zip(L_arr, omega_arr):
            res_mtx = self.signal_inductor_ABCD_func_(L, omega)
            res_mtx_arr.append(res_mtx)
        res_mtx_arr_np = np.array(res_mtx_arr, dtype='complex')
        return res_mtx_arr_np


    def evaluate_abcd_mtx(self, L_val, Jpa_val, omega_arr, active=True):
        '''
        returns the ABCD matrix of the network for each inductance, inverter coupling rate, and frequency in the input arrays
        This can only handle variation over frequency! The inductance and coupling rate must be a single constant
        '''
        if np.size(L_val) > 1 or np.size(Jpa_val) > 1:
            raise Exception("Improper variation in ABCD mtx eval, use only one inductance at a time")
        L_arr = L_val * np.ones_like(omega_arr)
        Jpa_arr = Jpa_val * np.ones_like(omega_arr)
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

        return abcd_to_s(self.evaluate_abcd_mtx(L_val, Jpa_val, omega_arr), 50, num=True)


    def compare_abcd_to_ideal(self, analytical_net):
        '''
        compares the ABCD matrix of the simulated network to the ideal ABCD matrix of the filter circuit as computed from the network_synthesis module
        :param analytical_net:
        :return:
        '''
        pass


    def find_p2_input_impedance(self, L_val, omega_arr, Z0=50):
        '''
        returns the impedance seen from the inverter
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_abcd_mtx(L_val, 0, omega_arr, active=False)
        self.filterZmtxs = abcd_to_z(ABCD_mtxs, num=True)
        Z = self.filterZmtxs
        # to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        # impedance matrix and the source impedance
        port2_input_impedance = Z[1, 1] - Z[1, 0] * Z[0, 1] / (Z[0, 0] + Z0)

        return port2_input_impedance


    def find_p1_input_impedance(self, L_val, omega_arr, Zl=50):
        '''
        returns the impedance seen from the environment
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_abcd_mtx(L_val, 0, omega_arr, active=False)
        self.filterZmtxs = abcd_to_z(ABCD_mtxs, num=True)
        Z = self.filterZmtxs
        # to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        # impedance matrix and the source impedance
        port1_input_impedance = Z[0, 0] - Z[1, 0] * Z[0, 1] / (Z[1, 1] + Zl)

        return port1_input_impedance


    def modes_as_function_of_inductance(self, l_arr, omega_arr, debug=False, maxiter=10000, Z0=50):
        '''
        Takes in an array of inductance values and frequencies
        returns the modes as a function of the inductance of the array inductor. In the format
        of the return of find_modes_from_input_impedance
        '''

        res_list = []
        res_params_list = []
        for l_val in l_arr:
            if debug: print("Inductance value: ", l_val * 1e12, " pH")
            z_arr = self.find_p2_input_impedance(l_val, omega_arr, Z0=Z0)
            res = find_modes_from_input_impedance(z_arr, omega_arr, debug=debug, maxiter=maxiter)
            res_params = mode_results_to_device_params(res)
            if len(res) != 0:
                res_list.append(res)
                res_params_list.append(res_params)
        return ModeResult(l_arr, res_list, res_params_list)


csv_str_to_complex = lambda array_x: np.array([complex(x.replace('i', 'j').replace(' ', '')) for x in array_x])


class NdHFSSSweepOptimizer:
    '''
    Most HFSS driven modal sweeps are parametrically changed versus multiple parameters, and we need to find a way to
    minimize error inside that parameter space. This class is designed to take in a single csv exported from HFSS
    that has scattering parameters plotted in the following format:
    first axis: parameter 1
    second axis: parameter 2
    ...
    nth axis: parameter n
    n+1th axis: frequency
    n+2th axis: Z11
    n+3th axis: Z12
    n+4th axis: Z21
    n+5th axis: Z22

    All of these parameters should be in complex format. Z parameters are used instead of S parameters
    so that our results are independent of the reference plane.

    The first port is the real one, representing the signal port. The second port is the one representing the inductor
    The flow overall is to create skrf network objects for each combination of parameters, then use those to construct
    inductor sweeps for each axis. From there, we can interpolate the inductor sweeps and calculate a cost function for
    each parameter combination. At the very least this should give us a way to find the best parameter combination.

    The second level is to interpolate the cost function landscape. This will allow us to find the best parameter
    combination within the space enclosed by the parameters we swept, instead of just the best point we tested. Ideally
    these two should be close, but this landscape may have many local minima.
    '''

    def __init__(self, filename):
        self.filename = filename

    def optimize_params(self, omega_arr=None):
        self.par_axes, self.combos, self.skrf_nets, self.HFSS_omega_arr = self.skrf_nets_from_sweep_file()
        if omega_arr is None:
            self.omega_arr = self.HFSS_omega_arr
        else:
            self.omega_arr = omega_arr
            print("HFSS freqs (GHz): ", self.HFSS_omega_arr / 2 / np.pi / 1e9,
                  "\nomega_arr/2/pi (GHz): ", omega_arr / 2 / np.pi / 1e9)
        # calculate the cost functions for each parameter combination
        self.cost_arr, self.z_vals_arr, self.ideal_z_vals_arr = self.calculate_cost_landscape(self.combos,
                                                                                              self.skrf_nets)
        self.find_test_cost_minimum(self.combos, self.cost_arr, self.ideal_z_vals_arr, self.z_vals_arr, self.par_axes)

    def skrf_nets_from_sweep_file(self):
        '''
        takes in a csv file and returns skrf files indexed by the parameters.
        The final format for this will be flattened into a 1D array, with [(par1, par2, ...], [skrf_files])
        '''
        sweep_file = pd.read_csv(self.filename)
        # build arrays that represent the axes of the csv file
        par_axes = [sweep_file.columns[i] for i in range(len(sweep_file.columns) - 5)]
        par_axes_vals = [sweep_file[par_axes[i]].unique() for i in range(len(par_axes))]
        freqs = np.unique(sweep_file[sweep_file.columns[-5]])
        # iterate through the axes values, constructing the 1D array as we go
        skrf_files = []
        # outermost layer: iterate over axes
        # construct all combinations of the independent parameter values in a 1d array using combinations module

        combos = np.array(np.meshgrid(*par_axes_vals)).T.reshape(-1, len(par_axes))
        nets = []
        print("Processing combinations of variables into SKRF networks...")
        for i, par_vals in tqdm(enumerate(combos)):
            filters = [sweep_file[par_axes[i]] == par_vals[i] for i in range(len(par_vals))]
            filtered_df = sweep_file
            for filt in filters:
                filtered_df = filtered_df[filt]
            # now we have the filtered df, we can construct the skrf file
            freqs = np.hstack(filtered_df[filtered_df.columns[-5:-4]].to_numpy())
            z11 = csv_str_to_complex(np.hstack(filtered_df[filtered_df.columns[-4:-3]].to_numpy()))
            z12 = csv_str_to_complex(np.hstack(filtered_df[filtered_df.columns[-3:-2]].to_numpy()))
            z21 = csv_str_to_complex(np.hstack(filtered_df[filtered_df.columns[-2:-1]].to_numpy()))
            z22 = csv_str_to_complex(np.hstack(filtered_df[filtered_df.columns[-1:]].to_numpy()))
            zmtx = np.moveaxis(np.array([[z11, z12], [z21, z22]]), -1, 0)
            # f = rf.Frequency.from_f(freqs)
            # breakpoint()
            net = rf.Network.from_z(zmtx, f=freqs, f_unit='ghz')
            nets.append(net)
        return par_axes, combos, nets, freqs * 1e9 * 2 * np.pi

    def calculate_net_cost(self, L_val, omega_arr,
                           sim_net: InterpolatedNetworkWithInverterFromSKRF,
                           analytic_net: Network):
        '''
        calculates the cost function for a given network at a given inductance value
        '''
        # first, we need to calculate the impedance of the network at the given inductance
        net_z_vals = sim_net.find_p2_input_impedance(L_val, omega_arr, Z0=50)
        # now we need to calculate the ideal impedance at the given inductance
        ideal_z_vals = analytic_net.analytical_impedance_to_numerical_impedance_from_array_inductance(
            analytic_net.passive_impedance_seen_from_inverter()
        )(omega_arr, L_val)
        # now we need to calculate the cost function
        # this cost function just minimizes the difference over the whole frequency range
        cost = np.sum(np.abs(net_z_vals - ideal_z_vals))

        # #this cost function minimizes the normalized difference at the resonant frequency, alongside the normed gradient
        # #and curvature
        # ffilt = np.argmin(np.abs(omega_arr - analytic_net.omega0_val))
        # ideal_val = ideal_z_vals.real
        # net_val = net_z_vals.real
        # ideal_grad = np.gradient(ideal_z_vals.real)
        # ideal_curv = np.gradient(ideal_grad)
        # net_grad = np.gradient(net_z_vals.real)
        # net_curv = np.gradient(net_grad)
        #
        # cost = np.sum(np.abs((net_val - ideal_val)/ideal_val)
        #                 # np.abs((net_grad - ideal_grad)/ideal_grad)
        #                 # np.abs((net_curv - ideal_curv)/ideal_curv)
        #               )

        return cost, net_z_vals, ideal_z_vals

    def calculate_cost_landscape(self, combos, skrf_nets):
        cost_arr = []
        z_vals_arr, ideal_z_vals_arr = [], []
        for combo, skrf_net in zip(combos, skrf_nets):
            print("combo: ", combo)
            L_sym = sp.symbols('L')
            inv_J_sym = sp.symbols('J')
            omega0_val = 2 * np.pi * 7e9
            dw = 0.075
            omega_arr = skrf_net.f * 2 * np.pi
            sim_net = InterpolatedNetworkWithInverterFromSKRF(skrf_net,
                                                              L_sym,
                                                              inv_J_sym,
                                                              omega0_val,
                                                              dw)
            # for each one of these networks, we need to calculate a cost function at the operating inductance
            # breakpoint()
            cost, net_z_vals, ideal_z_vals = self.calculate_net_cost(L_squid, omega_arr, sim_net, ideal_net)
            cost_arr.append(cost)
            z_vals_arr.append(net_z_vals)
            ideal_z_vals_arr.append(ideal_z_vals)
        return cost_arr, z_vals_arr, ideal_z_vals_arr

    def find_test_cost_minimum(self, combos, cost_arr, ideal_z_vals_arr, z_vals_arr, par_names):
        min_cost_idx = np.argmin(cost_arr)
        min_cost_combo = combos[min_cost_idx]
        min_cost_z_vals = z_vals_arr[min_cost_idx]
        min_cost_ideal_z_vals = ideal_z_vals_arr[min_cost_idx]
        print(f"Minimum cost is {cost_arr[min_cost_idx]} at {min_cost_combo}")
        # plot the impedance of the least costly combination against the ideal impedance
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].set_title(f"Optimal values for\n{par_names}\n{min_cost_combo}")
        axs[1].set_title(f"Optimal values for\n{par_names}\n{min_cost_combo}")
        axs[0].plot(self.omega_arr, min_cost_z_vals.real, label='Simulated, real')
        axs[1].plot(self.omega_arr, min_cost_z_vals.imag, label='Simulated, imag')
        axs[0].plot(self.omega_arr, min_cost_ideal_z_vals.real, label='Ideal, real')
        axs[1].plot(self.omega_arr, min_cost_ideal_z_vals.imag, label='Ideal, imag')
        [ax.legend() for ax in axs]

        plt.show()

#TODO: everything below here is deprecated

@dataclass
class InterpolatedNetworkWithInverterFromFilename:
    '''
    This class takes in an s2p file from HFSS and analyzes the results by finding modes and computing nonlinearities
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
        inverter_total_abcd = self.inverter.abcd_signal_inductor_shunt() * self.inverter.abcd_inverter_shunt() * self.inverter.abcd_idler_inductor_shunt()
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

    def modes_as_function_of_inductance(self, l_arr, omega_arr, debug=False, maxiter = 10000, Z0 = 50):
        '''
        Takes in an array of inductance values and frequencies
        returns the modes as a function of the inductance of the array inductor. In the format
        of the return of find_modes_from_input_impedance
        '''

        res_list = []
        res_params_list = []
        for l_val in l_arr:
            if debug: print("Inductance value: ", l_val * 1e12, " pH")
            z_arr = self.find_p2_input_impedance(l_val, omega_arr, Z0=Z0)
            res = find_modes_from_input_impedance(z_arr, omega_arr, debug=debug, maxiter = maxiter)
            res_params = mode_results_to_device_params(res)
            if len(res) != 0:
                res_list.append(res)
                res_params_list.append(res_params)
        return ModeResult(l_arr, res_list, res_params_list)


@dataclass
class InterpolatedNetworkWithoutInverterFromFilename:
    '''
    This file takes in an s2p file from HFSS and analyzes the results by finding modes and computing nonlinearities
    '''
    filename: str
    omega0_val: float
    dw: float

    def __post_init__(self):
        self.skrf_network = import_s2p(self.filename)

    def evaluate_abcd_mtx(self, omega_arr, active = True):
        '''
        returns the ABCD matrix of the network for each inductance, inverter coupling rate, and frequency in the input arrays
        This can only handle variation over frequency! The inductance and coupling rate must be a single constant
        '''

        # this will return a 2x2xN matrix of floats, with 1xN input arrays

        self.s2p_net_ABCD_mtx_signal = interpolate_mirrored_abcd_functions(self.skrf_network, omega_arr)

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

        # print("Debug: signal inductor ABCD array shape:", signal_inductor_ABCD_array.shape)
        total_ABCD_mtx_evaluated = self.s2p_net_ABCD_mtx_signal

        # now we have a total Nx2x2 ABCD matrix, but to convert that to a scattering matrix, we need the 2x2xN shape back
        # so we use moveaxis again
        self.total_ABCD_mtx_evaluated_reshaped = np.moveaxis(total_ABCD_mtx_evaluated, 0, -1)

        return self.total_ABCD_mtx_evaluated_reshaped

        # now we can convert to S parameters using the helper function I already have
    def evaluate_smtx(self, omega_arr):

        return abcd_to_s(self.evaluate_abcd_mtx(omega_arr), 50, num = True)

    def find_p2_input_impedance(self, omega_arr, Z0 = 50):
        '''
        returns the impedance seen from the inverter
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_abcd_mtx(omega_arr, active = False)
        self.filterZmtxs = abcd_to_z(ABCD_mtxs, num=True)
        Z = self.filterZmtxs
        #to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        #impedance matrix and the source impedance
        port2_input_impedance = Z[1,1]-Z[1,0]*Z[0,1]/(Z[0,0]+Z0)

        return port2_input_impedance

    def find_p1_input_impedance(self, omega_arr, Zl = 50):
        '''
        returns the impedance seen from the environment
        (as converted from the ABCD matrix)
        '''
        ABCD_mtxs = self.evaluate_abcd_mtx(omega_arr, active = False)
        self.filterZmtxs = abcd_to_z(ABCD_mtxs, num=True)
        Z = self.filterZmtxs
        #to get modes from BBQ, we need to have the full impedance as seen from the inverter, which you can get from the
        #impedance matrix and the source impedance
        port1_input_impedance = Z[0,0]-Z[1,0]*Z[0,1]/(Z[1,1]+Zl)

        return port1_input_impedance

    def modes(self, omega_arr, debug=False, maxiter = 10000, Z0 = 50):
        '''
        Takes in an array of inductance values and frequencies
        returns the modes as a function of the inductance of the array inductor. In the format
        of the return of find_modes_from_input_impedance
        '''

        res_list = []
        res_params_list = []
        l_arr = [0]
        for l_val in l_arr:
            if debug: print("Inductance value: ", l_val * 1e12, " pH")
            z_arr = self.find_p2_input_impedance(l_val, omega_arr, Z0=Z0)
            res = find_modes_from_input_impedance(z_arr, omega_arr, debug=debug, maxiter = maxiter)
            res_params = mode_results_to_device_params(res)
            if len(res) != 0:
                res_list.append(res)
                res_params_list.append(res_params)
        return ModeResult(l_arr, res_list, res_params_list)

@dataclass
class ModeResult:
    ivar: np.ndarray #the variable that was swept over
    res: list
    '''
    the modes found at each value of the ivar, this can have variable first dimension N, 
    but has second dimension M shaped from the result of find_modes_from_input_impedance
    '''
    res_params: list
    '''
    the mode parameters found at each value of the ivar, this can have variable first dimension N,
    but has second dimension M shaped from the result of mode_results_to_device_params
    '''
    def __post_init__(self):
        # shape of mode_results is NxM where N is number of parameters, M is number of modes
        # make everything into 1d arrays, then filter afterwards
        L_list = []
        omega_list = []
        Ceff_list = []
        Leff_list = []
        Zpeff_list = []
        Q_list = []

        for L, modes, mode_params in zip(self.ivar, self.res, self.res_params):
            mode_result_freqs = modes[0]
            # print(mode_result_freqs)
            mode_result_Qs = mode_params[0]
            mode_result_Cs = mode_params[1]
            mode_result_Ls = mode_params[2]
            # print(mode_result_Ls)
            mode_result_Zpeffs = mode_params[3]

            L_list.append(L * np.ones_like(mode_result_freqs))
            omega_list.append(mode_result_freqs)
            Ceff_list.append(mode_result_Cs)
            Leff_list.append(mode_result_Ls)
            Q_list.append(mode_result_Qs)
            Zpeff_list.append(mode_result_Zpeffs)

        self.ivar_arr = np.hstack(L_list)
        self.omega_arr = np.hstack(omega_list)
        self.Ceff_arr = np.hstack(Ceff_list)
        self.Leff_arr = np.hstack(Leff_list)
        self.Zpeff_arr = np.hstack(Zpeff_list)
        self.Q_arr = np.hstack(Q_list)
        # for i in range(self.ivar.size):
        #     if len(self.mode_res_arr[i][0]) != 0:
        #         self.ivar_cleaned.append(self.ivar[i])
        #         self.mode_res_arr_cleaned.append(np.array(self.mode_res_arr[i]))
        #         self.mode_params_arr_cleaned.append(np.array(self.mode_params_arr[i]))
        #         self.mode_res_dict[self.ivar[i]] = dict(
        #             zip(
        #             [i for i in range(len(self.mode_res_arr[i]))],
        #             np.array(self.mode_res_arr[i]).T
        #             )
        #         )
        #         self.mode_params_dict[self.ivar[i]] = dict(
        #             zip(
        #                 [i for i in range(len(self.mode_params_arr[i]))],#this gives each mode an index
        #                 np.array(self.mode_params_arr[i]).T
        #             )
        #         )