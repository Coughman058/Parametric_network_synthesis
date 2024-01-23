from .helper_functions import *
from .component_ABCD import *
from ..drawing_tools.sketching_functions import draw_net_by_type
import matplotlib.pyplot as plt
# from tensorwaves.function.sympy import fast_lambdify

def get_active_network_prototypes():
    """
    Returns a dictionary of the active network prototypes, with the keys being the names of the prototypes
    :return: dictionary of the active network prototypes
    """
    active_network_prototypes = dict(
        N3_Butter_20dB = np.array([1.0, 0.5846, 0.6073, 0.2981, 0.9045]),
        N3_Cheby_20dB_R01 = np.array([1.0, 0.4656, 0.5126, 0.2707, 0.9045]),
        N3_Cheby_20dB_R05 = np.array([1.0, 0.5899, 0.6681, 0.3753, 0.9045]),
        N2_Cheby_20dB_R05 = np.array([1.0, 0.3184, 0.1982, 1.1055]),
        N4_Leg_20dB_R05 = np.array([1.0, 0.6886, 0.8864, 0.8918, 0.2903, 1.1055]),
        N2_Leg_20dB_R05 = np.array([1.0, 0.3105, 0.1868, 1.1055])
    )
    return active_network_prototypes

def get_passive_network_prototypes():
    passive_network_prototypes = dict(
    )
    return passive_network_prototypes

def calculate_network(g_arr, z_arr, f0, dw, L_squid, printout=True):
    """
    Calculates the network parameters for a given set of g and z values.
    :param g_arr: array of g values, unique for each filter prototype
    :param z_arr: array of z values, these are free parameters, but the last one is the output impedance
    :param f0: center frequency, in Hz
    :param dw: fractional bandwidth
    :param L_squid: inductance of the squid
    :param printout: print the results or not
    :return: network object, containing the network parameters
    """

    w0 = 2 * np.pi * f0
    Z0 = z_arr[-1]
    C_squid = 1 / (w0 ** 2 * L_squid)
    ZPA_res = np.sqrt(L_squid / C_squid)
    Z_last = calculate_last_resonator_impedance(dw, Z0, g_arr[-1], g_arr[-2])
    # Z_last_lambda_over_2 = Z_last*np.pi/4
    # from PA out
    z_arr[0] = ZPA_res
    z_arr[-2] = Z_last
    # np.array([ZPA_res,20,20,Z_last,50])
    # z_arr = np.array([ZPA_res,Z_last,50])
    # now calculate the inverters
    J_arr = np.array(
        [calculate_middle_inverter_constant(dw, g_arr[i + 1], g_arr[i + 2], z_arr[i], z_arr[i + 1]) for i in
         range(len(g_arr) - 2)]
        #  calculate_middle_inverter_constant(dw, g_arr[2], g_arr[3], z_arr[1], z_arr[2]),
        #  calculate_middle_inverter_constant(dw, g_arr[3], g_arr[4], z_arr[2], z_arr[3]),
        #  calculate_middle_inverter_constant(dw, g_arr[4], g_arr[5], z_arr[3], z_arr[4]),
    )

    J_arr[-1] /= np.sqrt(dw)
    CC_arr = J_arr / w0
    CC_arr[-1] = 0
    CC_arr_padded = np.pad(CC_arr, 1)
    C_arr_uncomp = 1 / w0 / z_arr
    C_arr = np.array([C_arr_uncomp[i] - CC_arr_padded[i] - CC_arr_padded[i + 1] for i in range(len(C_arr_uncomp))])
    C_arr[-1] = 0
    L_arr = z_arr / w0
    L_arr[-1] = 0

    gamma_0 = dw / g_arr[-1] / g_arr[-2]

    beta_arr = np.array(
        [calculate_middle_beta(dw, g_arr[i + 1], g_arr[i + 2], gamma_0) for i in range(len(g_arr) - 2)]
    )

    beta_p = calculate_PA_beta(g_arr)

    if printout:
        print("ZPA_res", ZPA_res)
        print("J_arr: ", J_arr)
        print("lambda/4 impedance array: ", 1 / J_arr)
        print("CC_arr: ", CC_arr * 1e12, "pF")
        print("C_arr: ", C_arr * 1e12, "pF")
        print("C_arr_uncomp: ", C_arr_uncomp * 1e12, "pF")
        print("L_arr: ", L_arr * 1e9, "nH")
        print("Z_arr: ", z_arr)
        print("Active Resistance, ", calculate_PA_impedance(ZPA_res, g_arr[0], g_arr[1], dw))

    return network(omega0_val=w0,
                   g_arr=g_arr,
                   dw=dw,
                   J=J_arr,
                   CC=CC_arr,
                   C=C_arr,
                   Cu=C_arr_uncomp,
                   L=L_arr,
                   Z=z_arr,
                   beta=beta_arr,
                   beta_p=beta_p,
                   R_active_val=calculate_PA_impedance(ZPA_res, g_arr[0], g_arr[1], dw))


@dataclass
class network:

    """
    Class containing the network parameters for a given set of g and z values.
    :param omega0_val: center frequency, in rad/s
    :param g_arr: array of g values, unique for each filter prototype
    :param dw: fractional bandwidth
    :param J: array of J values, these are coupling rates
    :param CC: array of CC values, these are capacitances of the coupling capacitors
    :param C: array of C values, these are capacitances of the capacitors in the resonators
    :param Cu: array of C values, these are uncompensated capacitances of the capacitors in the resonators, used for calculating the inductances
    :param L: array of L values, these are inductances of the inductors in the resonators
    :param Z: array of Z values, these are impedances of the resonators
    :param beta: array of beta values, these are the beta values that connect the resonators in coupled-mode theory
    :param beta_p: beta value of the parametric inverter that sits in teh middle of a degenerate JPA design
    :param R_active_val: active resistance of the parametric inverter when using the pumpistor model

    """

    omega0_val: float
    g_arr: np.ndarray
    dw: float
    J: np.ndarray
    CC: np.ndarray
    C: np.ndarray
    Cu: np.ndarray
    L: np.ndarray
    Z: np.ndarray
    beta: np.ndarray
    beta_p: float
    R_active_val: float

    def __post_init__(self):
        self.Ftypes = [
            'cap_cpld_lumped',
            'tline_cpld_lumped',
            'tline_cpld_l4',
            'tline_cpld_l2',
            'ideal']
        self.net_size = self.J.size - 1

    def debug_printout(self):
        """
        Prints out the network parameters
        """
        print(self.Ftype)
        print(self.Z)
        print("if l4:")
        print(np.array(self.Z) * np.pi / 4)
        print("if l2:")
        print(np.array(self.Z) * np.pi / 2)
        print('net TLINE Zs: ')
        print([el.Zval for el in self.net_elements if el.__class__.__name__ == 'TLINE'])
        print('net TLINE Thetas: ')
        print([el.theta_val for el in self.net_elements if el.__class__.__name__ == 'TLINE'])
        print('net Cs: ')
        print([el.val for el in self.net_elements if el.__class__.__name__ == 'capacitor'])
        print('net Ls: ')
        print([el.val for el in self.net_elements if el.__class__.__name__ == 'inductor'])
        # [display(i, el.__class__.__name__) for (i, el) in enumerate(self.net_elements)]
        # [display(el) for (i, el) in enumerate(self.ABCD_mtxs)]
        # display(self.net_subs)

    def lumped_res(self, n: int, net_size, omega_sym: sp.Symbol, include_inductor=True, compensated=False, conjugate=False):
        """
        Adds a lumped resonator to the network
        :param n: location of the resonator in the network, this will be used to name the elements and assign the values
        to variables
        :param net_size: number of resonators in the network. This is actually unused in this function, but is used in
        the lumped_res_compensated function
        :param omega_sym: the symbol for the angular frequency
        :param include_inductor: whether to include an inductor in the resonator. This is used for the first resonator
        in the network, which has the parametric inverter element that includes the inductor
        :param compensated: whether the capactiors in the resonator are compensated by coupling caps or not
        :return:
        """
        if include_inductor:
            ind_symbol = sp.symbols(f'L_{n}', positive=True)
            ind_val = self.L[n]
            ind_el = inductor(omega_sym, ind_symbol, ind_val)
            self.net_elements.insert(0, ind_el)
            if conjugate:
                self.ABCD_mtxs.insert(0, sp.conjugate(ind_el.ABCDshunt()))
            else:
                self.ABCD_mtxs.insert(0, ind_el.ABCDshunt())
            self.net_subs.insert(0, (ind_symbol, ind_val))

        cap_symbol = sp.symbols(f'C_{n}', positive=True)
        if compensated:
            cap_val = self.C[n]
        else:
            cap_val = self.Cu[n]
        cap_el = capacitor(omega_sym, cap_symbol, cap_val)
        self.net_elements.insert(0, cap_el)
        if conjugate:
            self.ABCD_mtxs.insert(0, sp.conjugate(cap_el.ABCDshunt()))
        else:
            self.ABCD_mtxs.insert(0, cap_el.ABCDshunt())
        self.net_subs.insert(0, (cap_symbol, cap_val))
        Zres_symbol = sp.symbols(f"Zr_{n}", positive=True)
        omega_str = f"omega_r_{n}"
        omega_res_symbol = sp.symbols(omega_str, positive=True)
        self.parameter_subs += [(cap_symbol, 1 / (Zres_symbol * omega_res_symbol))]
        self.res_omega_symbols.append(omega_res_symbol)
        self.res_Z_symbols.append(Zres_symbol)
        if include_inductor: self.parameter_subs += [(ind_symbol, Zres_symbol / omega_res_symbol)]

    def tline_res(self, n, net_size, omega_sym, res_type='lambda4', use_approx=False, conjugate=False):

        """
        Adds a transmission line resonator to the network
        :param n: same as in lumped_res
        :param net_size:  same as in lumped_res
        :param omega_sym:  same as in lumped_res
        :param res_type: the type of TLINE resonator to add, either 'lambda4' or 'lambda2'
        :param use_approx: This is used if you want to truncate the trigonometric functions in the TLINE ABCD matrix
        to a certain order.
        :return:
        """

        res_types = ['lambda4_shunt', 'lambda2_shunt']
        # TODO: add 'lambda2_series'
        if res_type not in res_types:
            raise Exception("resonator type must be 'lambda4' or 'lambda2' ")

        tline_omega0_val = self.omega0_val
        tline_Z_symbol, tline_theta_symbol, tline_omega0_symbol = sp.symbols(f'Zr_{n}, theta_r_{n}, omega_r_{n}',
                                                                             positive=True)

        if res_type == 'lambda4_shunt':
            # print('shunt l4')
            tline_Z_val = self.Z[n] * (np.pi / 4)
            tline_theta_val = sp.pi / 2
            tline_el = TLINE(
                omega_sym,
                tline_Z_symbol,
                tline_theta_symbol,
                tline_omega0_symbol,
                tline_Z_val,
                tline_theta_val,
                tline_omega0_val)
            if conjugate:
                self.ABCD_mtxs.insert(0, sp.conjugate(tline_el.ABCDshunt_short(use_approx=use_approx)))
            else:
                self.ABCD_mtxs.insert(0, tline_el.ABCDshunt_short(use_approx=use_approx))

        elif res_type == 'lambda2_shunt':
            # print('shunt l2')
            tline_Z_val = self.Z[n] * (np.pi / 2)
            tline_theta_val = sp.pi
            tline_el = TLINE(
                omega_sym,
                tline_Z_symbol,
                tline_theta_symbol,
                tline_omega0_symbol,
                tline_Z_val,
                tline_theta_val,
                tline_omega0_val)
            if conjugate:
                self.ABCD_mtxs.insert(0, sp.conjugate(tline_el.ABCDshunt_open(use_approx=use_approx)))
            else:
                self.ABCD_mtxs.insert(0, tline_el.ABCDshunt_open(use_approx=use_approx))
        else:
            raise Exception('error in TLINE type')

        self.net_elements.insert(0, tline_el)
        self.net_subs.insert(0, (tline_theta_symbol, tline_theta_val))
        self.net_subs.insert(0, (tline_Z_symbol, tline_Z_val))
        self.net_subs.insert(0, (tline_omega0_symbol, tline_omega0_val))

    def cap_cpld_lumped_unit(self, n, net_size, omega_sym, include_inductor=True, conjugate=False):
        """
        Adds a lumped element resonator and a coupling capacitor in series to the network
        :param n: same as in lumped_res
        :param net_size: same as in lumped_res
        :param omega_sym: same as in lumped_res
        :param include_inductor: same as in lumped_res
        :return:
        """
        # resonator
        self.lumped_res(n, net_size, omega_sym,
                        include_inductor=include_inductor,
                        compensated=True, conjugate=conjugate)
        # coupler
        if n != net_size:  # all these have eliminated port inverters
            cpl_symbol = sp.symbols(f'Cc_{n}', positive=True)
            cpl_val = self.CC[n]
            cpl_el = capacitor(omega_sym, cpl_symbol, cpl_val)
            self.net_elements.insert(0, cpl_el)
            self.net_subs.insert(0, (cpl_symbol, cpl_val))
            if conjugate:
                self.ABCD_mtxs.insert(0, sp.conjugate(cpl_el.ABCDseries()))
            else: 
                self.ABCD_mtxs.insert(0, cpl_el.ABCDseries())

    def tline_cpld_lumped_unit(self, n, net_size, omega_sym,
                               include_inductor=True,
                               tline_inv_Z_corr_factor=1, use_approx=False, conjugate=False):
        '''
        Adds a lumped element resonator and a coupling capacitor in series to the network
        :param n: same as in lumped_res
        :param net_size: same as in lumped_res
        :param omega_sym: same as in lumped_res
        :param include_inductor: same as in lumped_res
        :param tline_inv_Z_corr_factor: This is used to correct the impedance of the transmission line resonator.
        This is because the impedance of a transmission line resonator is not the same as the characteristic impedance
        of the line it is made out of. Derivation here:
        https://colab.research.google.com/drive/15clNBBCLazeFt3JM8UBDtSIJNKeAENR_?usp=sharing
        '''
        # resonator
        self.lumped_res(n, net_size, omega_sym,
                        include_inductor=include_inductor,
                        compensated=False, conjugate=conjugate)
        # coupler
        if n != net_size:  # all these have eliminated port inverters

            tline_Z_symbol, tline_theta_symbol, tline_omega_symbol = sp.symbols(f'Z_{n}, theta_{n}, omega_{n}',
                                                                                positive=True)
            tline_Z_val = 1 / self.J[n] * tline_inv_Z_corr_factor
            tline_theta_val = sp.pi / 2
            tline_omega_val = self.omega0_val

            tline_el = TLINE(omega_sym,
                             tline_Z_symbol, tline_theta_symbol,
                             tline_omega_symbol,
                             tline_Z_val,
                             tline_theta_val,
                             tline_omega_val)

            self.net_elements.insert(0, tline_el)
            if conjugate:
                self.ABCD_mtxs.insert(0, sp.conjugate(tline_el.ABCDseries(use_approx=use_approx)))
            else:
                self.ABCD_mtxs.insert(0, tline_el.ABCDseries(use_approx=use_approx))
            self.net_subs.insert(0, (tline_theta_symbol, tline_theta_val))
            self.net_subs.insert(0, (tline_Z_symbol, tline_Z_val))
            self.net_subs.insert(0, (tline_omega_symbol, tline_omega_val))

    def tline_cpld_tline_unit(self, n, net_size, omega_sym,
                              tline_inv_Z_corr_factor=1,
                              tline_res_type='lambda4_shunt',
                              use_approx=False, conjugate=False):
        '''
        Adds a transmission line resonator and a lambda/4 inverter in series to the network
        :param n: same as in lumped_res
        :param net_size: same as in lumped_res
        :param omega_sym: same as in lumped_res
        :param tline_inv_Z_corr_factor: same as in tline_cpld_lumped_unit
        :param tline_res_type: same as in tline_cpld_lumped_unit
        :param use_approx: same as in tline_cpld_lumped_unit
        :return:
        '''
        # resonator
        self.tline_res(n, net_size, omega_sym,
                       res_type=tline_res_type, conjugate=conjugate)
        # coupler
        if n != net_size:  # all these have eliminated port inverters

            tline_Z_symbol, tline_theta_symbol, tline_omega_symbol = sp.symbols(f'Z_{n}, theta_{n}, omega_{n}',
                                                                                positive=True)
            tline_Z_val = 1 / self.J[n] * tline_inv_Z_corr_factor
            tline_theta_val = sp.pi / 2
            tline_omega_val = self.omega0_val

            tline_el = TLINE(omega_sym,
                             tline_Z_symbol, tline_theta_symbol,
                             tline_omega_symbol,
                             tline_Z_val,
                             tline_theta_val,
                             tline_omega_val)

            self.net_elements.insert(0, tline_el)
            self.ABCD_mtxs.insert(0, tline_el.ABCDseries(use_approx=use_approx))
            self.net_subs.insert(0, (tline_theta_symbol, tline_theta_val))
            self.net_subs.insert(0, (tline_Z_symbol, tline_Z_val))
            self.net_subs.insert(0, (tline_omega_symbol, tline_omega_val))

    def circuit_unit(self, Ftype, n, net_size, omega_sym,
                     include_inductor=True, tline_inv_Z_corr_factor=1, use_approx=False, conjugate = False):
        """
        Adds a circuit unit to the network, this is a unit that is made out of a resonator and a coupler.
        Makes me wish for switch cases in python.
        :param Ftype: type of circuit unit, choose from 'cap_cpld_lumped', 'tline_cpld_lumped', 'tline_cpld_tline'
        :param n: same as in lumped_res
        :param net_size: same as in lumped_res
        :param omega_sym: same as in lumped_res
        :param include_inductor: same as in lumped_res
        :param tline_inv_Z_corr_factor: same as in tline_cpld_lumped_unit
        :param use_approx: same as in tline_cpld_lumped_unit
        :return:
        """
        Ftype = Ftype.lower()
        if Ftype not in self.Ftypes:
            raise Exception(f"type not incorporated, choose from {self.Ftypes}")

        if Ftype == 'cap_cpld_lumped':
            self.cap_cpld_lumped_unit(n, net_size, omega_sym,
                                      include_inductor=include_inductor, conjugate = conjugate)
        elif Ftype == 'tline_cpld_lumped' or n == 0:
            self.tline_cpld_lumped_unit(
                n, net_size, omega_sym,
                include_inductor=include_inductor,
                tline_inv_Z_corr_factor=tline_inv_Z_corr_factor,
                use_approx=use_approx,
                conjugate = conjugate)

        elif Ftype == 'tline_cpld_l4':
            self.tline_cpld_tline_unit(
                n, net_size, omega_sym,
                tline_res_type='lambda4_shunt',
                tline_inv_Z_corr_factor=tline_inv_Z_corr_factor,
                use_approx=use_approx,
                conjugate = conjugate)

        elif Ftype == 'tline_cpld_l2':
            self.tline_cpld_tline_unit(
                n, net_size, omega_sym,
                tline_res_type='lambda2_shunt',
                tline_inv_Z_corr_factor=tline_inv_Z_corr_factor,
                use_approx=use_approx,
                conjugate = conjugate
            )
        else:
            raise Exception('error in circuit_unit filter type')

    def gen_net_by_type(self, Ftype, active=True, core_inductor=False, method='pumpistor', tline_inv_Z_corr_factor=1,
                        use_approx=True, draw=True):
        self.Ftype = Ftype
        self.net_elements = []
        self.net_subs = []
        self.parameter_subs = []  # these are for analysis, and they convert resonators to LC combinations if they are lumped
        self.res_omega_symbols = []
        self.res_Z_symbols = []
        self.ABCD_mtxs = []
        self.Z0 = sp.symbols('Z0')
        self.name = Ftype
        self.tline_inv_Z_corr_factor = tline_inv_Z_corr_factor

        '''
        all these elements are ordered from the port outward.
        The core resonator is last.
        Then if you have it set to active, it will
        # 1.) remove the inductor shunt ABCD from the ABCD matrix array
        2.) reverse the array
        3.) add the inverter ABCD mtx to the beginning,
        4.) apply the same function but with w_s-w_p
        # 5.) remove the inductor again (the inverter ABCD mtx serves this purpose)
        6.) reverse the array again to restore the original order
        '''

        inv_ind_sym, alpha, phi, R_active = sp.symbols('L_{nl}, alpha, phi, R_{active}', real=True)
        signal_omega_sym = sp.symbols('omega_s', real=True)
        idler_omega_sym = sp.symbols('omega_i', real=True)
        self.signal_omega_sym = signal_omega_sym
        self.idler_omega_sym = idler_omega_sym

        net_size = self.J.size - 1
        for n in range(net_size + 1):
            if n == 0:
                self.circuit_unit(Ftype, n, net_size, signal_omega_sym,
                                  include_inductor=core_inductor,
                                  tline_inv_Z_corr_factor=tline_inv_Z_corr_factor,
                                  use_approx=use_approx)
            else:
                self.circuit_unit(Ftype, n, net_size, signal_omega_sym,
                                  include_inductor=True,
                                  tline_inv_Z_corr_factor=tline_inv_Z_corr_factor,
                                  use_approx=use_approx)
        #   class DegenerateParametricInverter_Amp:
        # omega0_val: float
        # L: sp.Symbol
        # R_active: sp.Symbol
        # w: float
        # g_arr: np.ndarray

        if active == True:
            Jpa_sym = sp.symbols("J_{pa}")
            R_active = sp.symbols('R_{pump}')
            self.inv_el = DegenerateParametricInverter_Amp(
                omega0_val=self.omega0_val,
                omega1=signal_omega_sym,
                omega2=idler_omega_sym,
                Jpa_sym=Jpa_sym,
                L=inv_ind_sym,
                R_active=R_active
            )
            if method == 'pumped_mutual':
                self.net_elements.append(self.inv_el.signal_inductor)
                self.net_elements.append(self.inv_el)
                self.net_elements.append(self.inv_el.idler_inductor)

                self.ABCD_mtxs.append(self.inv_el.ABCD_signal_inductor_shunt())
                self.ABCD_mtxs.append(self.inv_el.ABCD_inverter_shunt())
                self.ABCD_mtxs.append(self.inv_el.ABCD_idler_inductor_shunt())


                [l.reverse() for l in [self.net_elements, self.ABCD_mtxs]]
                for n in range(net_size + 1):
                    if n == 0:
                        self.circuit_unit(Ftype, n, net_size, idler_omega_sym, include_inductor=core_inductor,
                                          tline_inv_Z_corr_factor=tline_inv_Z_corr_factor,
                                          use_approx=use_approx, conjugate = False)
                    else:
                        self.circuit_unit(Ftype, n, net_size, idler_omega_sym, include_inductor=True,
                                          tline_inv_Z_corr_factor=tline_inv_Z_corr_factor,
                                          use_approx=use_approx, conjugate = False)

                [l.reverse() for l in [self.net_elements, self.ABCD_mtxs]]

            elif method == 'pumpistor':

                self.net_elements.append(self.inv_el)
                self.ABCD_mtxs.append(self.inv_el.ABCD_pumpistor().subs(alpha, 0))

            if draw == True:
                draw_net_by_type(self, self.Ftype)

    def inverter_no_detuning_subs(self, omega: sp.Symbol):
        """
        Generates the substitution dictionary for an inverter with no detuning
        :param omega: sympy symbol for the angular frequency
        :return: list of tuples of the form (symbol, value) for the .subs() method
        """
        self.omega_from_inverter = omega
        return [(self.Z0, 50),
                # (self.inv_el.phi, 0),
                (self.inv_el.omega1, omega),
                (self.inv_el.omega2, omega - 2 * self.omega0_val),
                (self.inv_el.L, self.L[0])
                ]

    def calculate_Smtx(self, Z0):
        """
        Calculates the scattering matrix for the network
        :param Z0: characteristic impedance of the environment
        :return:
        """
        total_ABCD = self.total_ABCD()
        Smtx = ABCD_to_S(total_ABCD, Z0)
        return Smtx

    def calculate_ABCD(self):
        return self.total_ABCD()

    def plot_scattering(self, f_arr_GHz, additional_net_subs=[],
                        fig=None,
                        linestyle='solid',
                        primary_color='k',
                        secondary_color='grey',
                        label_prepend='',
                        vary_pump=True,
                        method='pumpistor',
                        focus=True,
                        debug = False):
        omega = sp.symbols('omega')
        if debug: print('Substituting all network values into component ABCD mtxs...')
        self.net_subs += self.inverter_no_detuning_subs(omega) + additional_net_subs
        self.plot_ABCD_mtxs = [ABCD.subs(self.net_subs) for ABCD in self.ABCD_mtxs]
        if debug: print('Calculating symbolic scattering matrix...')
        # Smtx = self.calculate_Smtx(self.Z0)
        if debug: print('Calculating numerical scattering matrix...')
        SmtxN = ABCD_to_S(compress_ABCD_array(self.plot_ABCD_mtxs), 50)
        if debug: print('plotting results...')
        if method == 'pumpistor':
            Smtx_func = sp.lambdify([omega, self.inv_el.R_active], SmtxN)
        elif method == 'pumped_mutual':
            Smtx_func = sp.lambdify([omega, self.inv_el.Jpa_sym],SmtxN)
        omega_arr = f_arr_GHz * 2 * np.pi * 1e9

        net_size = np.size(self.g_arr) - 2
        if net_size % 2 == 0:
            j0val = self.dw / self.Z[0] / self.g_arr[1] / np.sqrt(self.g_arr[0]) * np.sqrt(self.g_arr[-1])
        else:
            j0val = self.dw / self.Z[0] / self.g_arr[1] / np.sqrt(self.g_arr[0]) / np.sqrt(self.g_arr[-1])

        if fig == None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        if vary_pump:
            Rvals = np.linspace(-1000, -self.R_active_val, 5)
            Jvals = np.arange(j0val / 2, 3 / 2 * j0val, j0val / 8)
        else:
            Rvals = np.array([-self.R_active_val])
            Jvals = np.array([j0val])
        if method == 'pumpistor':
            for i, Rval in enumerate(Rvals):
                alpha_val = Rval
                plt_Smtxs = Smtx_func(omega_arr, alpha_val)
                if focus:
                    if i == len(Rvals) - 1:
                        color = primary_color
                    else:
                        color = secondary_color
                else:
                    color = None
                ax.plot(f_arr_GHz, 20 * np.log10(np.abs(plt_Smtxs[0, 0])),
                        label=label_prepend + self.name + '\nR=-' + str(np.round(alpha_val, 1)),
                        linestyle=linestyle,
                        color=color
                        )
        elif method == 'pumped_mutual':
            for i, Jval in enumerate(Jvals):
                plt_Smtxs = Smtx_func(omega_arr, Jval)
                if focus:
                    if np.round(Jval, 4) == np.round(j0val, 4):
                        color = primary_color
                    else:
                        color = secondary_color
                else:
                    color = None
                ax.plot(f_arr_GHz, 20 * np.log10(np.abs(plt_Smtxs[0, 0])),
                        label=label_prepend + self.name + '\nJ=' + str(np.round(Jval, 4)),
                        linestyle=linestyle,
                        color=color
                        )
        return fig



    def total_ABCD(self):
        '''
        calculates ABCD matrix of the entire network thus far
        '''
        return compress_ABCD_array(self.ABCD_mtxs)

    def total_passive_ABCD(self, array = True):
        '''
        Let's calculate the scattering parameters of the network when the JPA is
        off. This should be easy, we're just looking at the phase structure of the
        network, the ABCD matrices are all multiplied together, and then transformed
        to scattering matrices
        '''
        #find out where the hell the inverter is. It could be just about anywhere depending on the network topology
        inverter_index = [(i, el) for (i, el) in enumerate(self.net_elements) if type(el) == DegenerateParametricInverter_Amp][0][0]

        if array:
            return compress_ABCD_array(self.ABCD_mtxs[0:inverter_index])
        else:
            return compress_ABCD_array(self.ABCD_mtxs[0:inverter_index-1])

    def passive_impedance_seen_from_array(self):
        '''
        This function calculates the impedance seen from the array port
        of the network, without including the array inductance
        '''
        ABCD = self.total_passive_ABCD(array = True)
        Z = ABCD_to_Z(ABCD, self.Z0)
        return Z[1,1]-Z[0,1]*Z[1,0]/(Z[0,0]+self.Z0)

    def passive_impedance_seen_from_inverter(self):
        '''
        This function calculates the impedance seen from the array port
        of the network inclding the array inductance
        '''
        ABCD = self.total_passive_ABCD(array = False)

        Z = ABCD_to_Z(ABCD, self.Z0)

        return Z[1, 1]-Z[0, 1]*Z[1, 0]/(Z[0, 0]+self.Z0)


    # def passive_impedance_seen_from_inverter(self):
    #     '''
    #     This function calculates the impedance seen from the input port
    #     of the network, and includes the array inductance
    #     '''
    #     ABCD = self.total_ABCD()
    #     return ABCD[0, 1] / ABCD[1, 1]

    # def evaluate_passive_ABCD_mtx_num(self):
    #     '''
    #     This function evaluates the ABCD matrices numerically
    #     '''
    #     self.ABCD_mtx_func = self.ABCD.subs(self.net_subs) for ABCD in self.ABCD_mtxs]
    #     return self.ABCD_mtxs_num

    # def compressed_active_ABCD_array(self, debug=True, method='pumped_mutual'):
    #     '''
    #     This function is deigned to compress the ABCD matrices of the signal and
    #     idler sides of the network without touching the inverter
    #     so that other analysis can be easier
    #     '''
    #     LHS_ABCD = sp.eye(2)
    #     net_size = len(self.ABCD_mtxs)
    #     if method.lower() == 'pumped_mutual':
    #         for i, ABCD in enumerate(self.ABCD_mtxs[0:net_size // 2]):
    #             if debug: print("LHS step ", i)
    #             LHS_ABCD *= ABCD
    #             LHS_ABCD = sp.simplify(LHS_ABCD)
    #
    #         self.ABCD_mtxs.reverse()
    #
    #         RHS_ABCD = sp.eye(2)
    #         for i, ABCD in enumerate(self.ABCD_mtxs[0:net_size // 2]):
    #             if debug: print("RHS step ", i)
    #             RHS_ABCD *= ABCD
    #             RHS_ABCD = sp.simplify(RHS_ABCD)
    #
    #         self.ABCD_mtxs.reverse()
    #
    #         return [LHS_ABCD, self.ABCD_mtxs[net_size // 2], RHS_ABCD]
    #
    #     if method.lower() == 'pumpistor':
    #         for i, ABCD in enumerate(self.ABCD_mtxs):
    #             if debug: print("LHS step ", i)
    #             LHS_ABCD *= ABCD
    #             LHS_ABCD = sp.simplify(LHS_ABCD)
    #
    #             return [LHS_ABCD, self.ABCD_mtxs[net_size - 1]]