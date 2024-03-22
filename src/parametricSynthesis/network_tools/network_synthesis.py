from .helper_functions import *
from .component_ABCD import *
from ..drawing_tools.sketching_functions import draw_net_by_type
import matplotlib.pyplot as plt
from tensorwaves.function.sympy import fast_lambdify
from ..simulation_tools.Quantizer import find_modes_from_input_impedance, mode_results_to_device_params
from scipy.optimize import newton
from tqdm import tqdm

def get_active_network_prototypes():
    """
    Returns a dictionary of the active network prototypes, with the keys being the names of the prototypes
    :return: dictionary of the active network prototypes
    """
    active_network_prototypes = dict(
        N2_Butter_20dB=np.array([ 1.0, 0.4085, 0.2343, 1.1055]),
        N2_Cheby_20dB_R05 = np.array([1.0, 0.3184, 0.1982, 1.1055]),
        N2_Leg_20dB_R05 = np.array([1.0, 0.3105, 0.1868, 1.1055]),
        N3_Butter_20dB = np.array([1.0, 0.5846, 0.6073, 0.2981, 0.9045]),
        N3_Cheby_20dB_R01 = np.array([1.0, 0.4656, 0.5126, 0.2707, 0.9045]),
        N3_Cheby_20dB_R05 = np.array([1.0, 0.5899, 0.6681, 0.3753, 0.9045]),
        N3_Leg_20dB_R01 = np.array([1.0, 0.4084, 0.4399, 0.2250, 0.9045]),
        N3_Leg_20dB_R05 = np.array([1.0, 0.5244, 0.5778, 0.3055, 0.9045]),
        N4_Leg_17dB = np.array([1.0, 0.9598, 1.1333, 1.3121, 0.4440, 1.1528]),
        N4_Leg_20dB_R05 = np.array([1.0, 0.6886, 0.8864, 0.8918, 0.2903, 1.1055])
    )
    return active_network_prototypes


def get_passive_network_prototypes():
    passive_network_prototypes = dict(
    )
    return passive_network_prototypes

def cap_to_tline_length_mod_factor(cap,
                                   z0,
                                   omega0):
    length_factor = (1-z0*omega0*cap*2/np.pi)
    print("compensating z = ", z0, "tline with cap = ", cap, "length factor = ", length_factor)
    return length_factor

def calculate_network(power_G_db, g_arr, z_arr, f0, dw, L_squid, printout=True, inv_corr_factor=1):
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
    if z_arr[-2] == 0:
        print("Last inverter will be eliminated")
        z_arr[-2] = Z_last
        elim_inverter = True
    else:
        print("Last inverter will be included")
        elim_inverter = False

    dw_limit = z_arr[-2]/z_arr[-1]*g_arr[-1]*g_arr[-2]
    first_res_lower_bound = dw*Z0/(g_arr[-1]*g_arr[-2])

    #sympy with the assist in finding the roots
    num_modes = len(g_arr) - 2
    #these indices mean nothing, look at the colab for more info
    g3_sp, g2_sp, g1_sp, J32_sp, J21_sp, Z3_sp, Z2_sp, Z1_sp = sp.symbols("g3 g2 g1 J32 J21 Z3 Z2 Z1")
    Js_sp = [('J%d%d' % (j, k), sp.symbols('J%d%d' % (j, k), positive=True)) for k in range(num_modes + 2) for j in
          range(num_modes + 2)]
    Zs_sp = [('Z%d' % j, sp.symbols('Z%d' % j, positive=True)) for j in range(num_modes + 2)]
    dw_sp, omega_sp, Zc_sp = sp.symbols('\delta\omega omega Zc')

    J32_rule = [(J32_sp, sp.sqrt(dw / (g3_sp * g2_sp * Z3_sp * Z2_sp)))]
    J21_rule = [(J21_sp, sp.sqrt(dw ** 2 / (g2_sp * g1_sp * Z2_sp * Z1_sp)))]
    [display(sp.Eq(*rule)) for rule in J32_rule + J21_rule]
    # print("The following must be less than 0, so let's find its roots:")
    expr = J32_sp * sp.sqrt(1 - (J32_sp * Z3_sp) ** 2) + J21_sp
    solveExpr = sp.simplify(
        (1 / expr.subs(J32_rule + J21_rule)) - Z2_sp)

    num_subs = [(Z3_sp, z_arr[-1]), (Z1_sp, z_arr[-3]), (g3_sp, g_arr[-1]), (g2_sp, g_arr[-2]), (g1_sp, g_arr[-3]),
                (omega_sp, 2 * np.pi * 7e9), (dw_sp, 0.1)]
    solveEq_num = sp.lambdify([Z2_sp], solveExpr.subs(num_subs))
    try:
        first_res_upper_bound = newton(solveEq_num, np.sqrt(z_arr[-3]*z_arr[-1]), maxiter = 10000)
    except RuntimeError:
        print("unable to find resonator upper bound")
        first_res_upper_bound = 1e10
    print("first resonator impedance must be between {} and {}".format(first_res_lower_bound,first_res_upper_bound))

    J_arr = np.array(
        [calculate_middle_inverter_constant(dw, g_arr[i + 1], g_arr[i + 2], z_arr[i], z_arr[i + 1]) for i in
         range(len(g_arr) - 2)]
    )
    # J_arr[0] /= np.sqrt(dw)
    J_arr[-1] /= np.sqrt(dw)
    J_arr*=inv_corr_factor
    CC_arr_raw = J_arr / w0
    if elim_inverter:
        CC_arr_raw[-1] = 0
        CC_arr = np.copy(CC_arr_raw)
        CC_arr_padded = np.pad(CC_arr, 1)
        C_arr_uncomp = 1 / w0 / z_arr
        C_arr = np.array([C_arr_uncomp[i] - CC_arr_padded[i] - CC_arr_padded[i + 1]
                          for i in range(len(C_arr_uncomp))])

    else:
        mod_factor = (1-z_arr[-1]**2*J_arr[-1]**2)
        if mod_factor<0:
            print(f"Capacitor modification <0, \nJ = {J_arr[-1]}, Z = {z_arr[-1]}"
                            "")

        CC_arr = np.copy(CC_arr_raw)
        CC_arr[-1] = CC_arr[-1]/np.sqrt(mod_factor)
        CC_arr_comp = np.pad(CC_arr, 1)
        CC_arr_comp[-2] = CC_arr_comp[-2]*mod_factor
        C_arr_uncomp = 1 / w0 / z_arr
        C_arr = np.array([C_arr_uncomp[i] - CC_arr_comp[i] - CC_arr_comp[i + 1]
                          for i in range(len(C_arr_uncomp))])
        tline_theta_arr_uncomp = np.array([np.pi/2 for i in range(len(C_arr_uncomp))])
        tline_theta_arr = np.array([tline_theta_arr_uncomp[i]*cap_to_tline_length_mod_factor(
            CC_arr_comp[i] + CC_arr_comp[i + 1],
            z_arr[i]*np.pi/4,
            w0) for i in range(len(C_arr_uncomp))])


    if np.any(C_arr<0):
        print("Warning: negative capacitance in filter, maybe try TLINE implementation")

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

    return Network(
        omega0_val=w0,
        g_arr=g_arr,
        dw=dw,
        J=J_arr,
        C=C_arr,
        Z=z_arr,
        power_G_db=power_G_db
    )
    # return Network(omega0_val=w0,
    #                g_arr=g_arr,
    #                dw=dw,
    #                J=J_arr,
    #                CC=CC_arr,
    #                C=C_arr,
    #                Cu=C_arr_uncomp,
    #                theta = tline_theta_arr,
    #                theta_u = tline_theta_arr_uncomp,
    #                L=L_arr,
    #                Z=z_arr,
    #                beta=beta_arr,
    #                beta_p=beta_p,
    #                R_active_val=calculate_PA_impedance(ZPA_res, g_arr[0], g_arr[1], dw),
    #                elim_inverter = elim_inverter)




@dataclass
class Network:
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
    C: np.ndarray
    Z: np.ndarray
    power_G_db: float

    def __post_init__(self):
        self.coupler_dict = {
            'cap': CapCoupler,
            'l4': TlineCoupler
        }
        self.resonator_dict = {
            'core': CoreResonator,
            'l4': TlineL4Resonator,
            'lumped': LumpedResonator}
        self.net_size = self.J.size - 1
        self.signal_omega_sym = sp.symbols('omega_s')
        self.idler_omega_sym = sp.symbols('omega_i')



    def gen_net_by_type(self, resonator_type_list, coupler_type_list, inv_corr_factors, draw=True):
        '''
        all these elements are ordered from the port inward.
        The core resonator is last.
        '''
        self.coupler_types = [s.lower() for s in coupler_type_list]
        self.resonator_types = [s.lower() for s in resonator_type_list]
        assert self.resonator_types[0] == 'core'

        self.ABCD_methods = []
        self.circuit_elements = []

        net_size = self.J.size - 1

        self.omega_0_sym = sp.symbols('omega_0')
        self.omega_s_sym = sp.symbols('omega_s')
        self.omega_i_sym = sp.symbols('omega_i')
        self.z_core_sym = sp.symbols('Z_{core}')
        self.jpa_sym = sp.symbols('J_pa')

        #this loop will construct the circuit and compile ABCD methods to be called by the scattering and impedance methods
        for n in range(2*net_size + 1):
            if n <= net_size:
                resonator_type = resonator_type_list[n]
                coupler_type = coupler_type_list[n]
            else:
                resonator_type = resonator_type_list[n-net_size]
                coupler_type = coupler_type_list[n-net_size]

            if resonator_type not in self.resonator_dict.keys():
                raise Exception("Resonator type not recognized")
            if coupler_type not in self.coupler_dict.keys():
                raise Exception("Coupler type not"
                                " recognized")

            if n == 0:
                #the core resonator has different arguments than the other resonators
                # start with the core resonator and coupler
                core_res = CoreResonator(
                    n=0,
                    omega_s_sym = self.omega_s_sym,
                    omega_i_sym = self.omega_i_sym,
                    omega0_sym = self.omega_0_sym,
                    omega0_val = self.omega0_val,
                    z_sym = self.z_core_sym,
                    z_val = self.Z[0],
                    jpa_sym = self.jpa_sym,
                    power_G_db = self.power_G_db,
                    g_arr = self.g_arr,
                    net_size = net_size,
                    dw = self.dw
                )

                core_coupler_s = self.coupler_dict[self.coupler_types[0]](
                    n=0,
                    omega_sym = self.omega_s_sym,
                    omega0_sym = self.omega_0_sym,
                    omega0_val = self.omega0_val,
                    j_val = self.J[0]*inv_corr_factors[0],
                    signal_or_idler_flag = 'signal'
                )
                core_coupler_i = self.coupler_dict[self.coupler_types[0]](
                    n=0,
                    omega_sym=self.omega_s_sym,
                    omega0_sym=self.omega_0_sym,
                    omega0_val=self.omega0_val,
                    j_val=self.J[0]*inv_corr_factors[0],
                    signal_or_idler_flag='idler'
                )
                self.circuit_elements.insert(0, core_coupler_i)
                self.circuit_elements.insert(0, core_res)
                self.circuit_elements.insert(0, core_coupler_s)
            elif n<=net_size:
                resonator_class = self.resonator_dict[resonator_type]
                coupler_class = self.coupler_dict[coupler_type]
                z_sym = sp.symbols('Z_' + str(n))
                print("coupler class: ", coupler_class.__name__)
                print("resonator class: ", resonator_class.__name__)
                res = resonator_class(
                    n = n,
                    omega_sym = self.omega_s_sym,
                    omega0_sym = self.omega_0_sym,
                    omega0_val = self.omega0_val,
                    z_sym = z_sym,
                    z_val = self.Z[n],
                    signal_or_idler_flag = 'signal'
                )
                coupler = coupler_class(
                    n = n,
                    omega_sym = self.omega_s_sym,
                    omega0_sym = self.omega_0_sym,
                    omega0_val = self.omega0_val,
                    j_val = self.J[n]*inv_corr_factors[n],
                    signal_or_idler_flag='signal'
                )
                self.circuit_elements.insert(0, res)
                self.circuit_elements.insert(0, coupler)
            elif n>net_size:
                resonator_class = self.resonator_dict[resonator_type]
                coupler_class = self.coupler_dict[coupler_type]
                z_sym = sp.symbols('Z_' + str(n))

                res = resonator_class(
                    n=n,
                    omega_sym=self.omega_s_sym,
                    omega0_sym=self.omega_0_sym,
                    omega0_val=self.omega0_val,
                    z_sym=z_sym,
                    z_val=self.Z[n-net_size],
                    signal_or_idler_flag='idler'
                )
                coupler = coupler_class(
                    n=n,
                    omega_sym=self.omega_s_sym,
                    omega0_sym=self.omega_0_sym,
                    omega0_val=self.omega0_val,
                    j_val=self.J[n-net_size]*inv_corr_factors[n-net_size],
                    signal_or_idler_flag = 'idler'
                )
                self.circuit_elements.append(res)
                self.circuit_elements.append(coupler)

            print("Circuit elements at ", n, ":", [el.__class__.__name__ for el in self.circuit_elements])
        #compensate the last coupler for a real termination on one side
        self.circuit_elements[0].compensate_end_capacitor(50)
        self.circuit_elements[0].synthesize()
        self.circuit_elements[-1].compensate_end_capacitor(50)
        self.circuit_elements[-1].synthesize()

        #now that all the elements are concatenated, we can compensate all the resonators
        for n, el in enumerate(self.circuit_elements):
            if el.__class__.__name__ == 'CoreResonator':
                el.compensate_for_couplers(self.circuit_elements[n - 1])
            elif el.__class__.__bases__[0].__name__ == 'Resonator':
                # print("compensating resonator at ", n)
                # print("name of resonator: ", el.__class__.__name__)
                el.compensate_for_couplers(self.circuit_elements[n-1], self.circuit_elements[n+1])

        #now we can synthesize all the rest of the elements
        for el in self.circuit_elements[1:-1]:
            el.synthesize()
            # print("Synthesized ", el.__class__.__name__)
        #and finally get all the abcd methods
        for el in self.circuit_elements:
            self.ABCD_methods.append(el.abcd_function)#this automatically gets the order right

    #now we need to create a function that can get the total ABCD matrix of the network
    def total_ABCD_func(self, omega_s, omega_i):
        print("Generating ABCD Matrices...")
        self.ABCD_mtxs_vs_frequency = [abcd(omega_s, omega_i) for abcd in self.ABCD_methods]
        # print("ABCD matrices vs frequency: ", self.ABCD_mtxs_vs_frequency)

        return compress_abcd_numerical(self.ABCD_mtxs_vs_frequency) #omega_s just for length

    def draw_circuit(self, l = 1.5):
        with schemdraw.Drawing() as d:
            d += elm.Ground()
            d += elm.RBox(label="$Z_0$").up()
            for el in self.circuit_elements[:len(self.circuit_elements)//2+1]:
                el.add_to_drawing(d, l = l)
        return d

    def scattering_from_inv_core_factors(self, inv_corr_factors):
        self.gen_net_by_type(self, self.resonator_types, self.coupler_types, inv_corr_factors)

        f_p_GHz = self.omega0_val/1e9/2/np.pi*2
        f_arr_GHz = self.omega_0_val/1e9/2/np.pi+np.linspace(-self.dw/2, self.dw/2, 1001)

        self.omega_s_arr = 2 * np.pi * f_arr_GHz * 1e9
        self.omega_i_arr = (2 * np.pi * f_arr_GHz * 1e9 - 2 * np.pi * f_p_GHz * 1e9)

        self.Smtx_j0 = self.Smtx_func(self.omega_s_arr, self.omega_i_arr)
    def optimize_inv_core_factors(self, omega_s, omega_i):
        '''
        This function optimizes inverter corrections using the network gain and fractional bandwidth, then
        returns the optimized inverter correction factors
        '''

    def Smtx_func(self, omega_s, omega_i):
        ABCD_mtx = np.moveaxis(self.total_ABCD_func(omega_s, omega_i), 0, -1)
        return abcd_to_s(ABCD_mtx, 50)

    def plot_scattering(self, f_arr_GHz, f_p_GHz,
                        fig=None,
                        linestyle='solid',
                        primary_color='k',
                        label_prepend=''):

        if fig == None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]

        self.omega_s_arr = 2 * np.pi * f_arr_GHz * 1e9
        self.omega_i_arr = (2 * np.pi * f_arr_GHz * 1e9 - 2 * np.pi * f_p_GHz *1e9)
        self.Smtx_j0 = self.Smtx_func(self.omega_s_arr, self.omega_i_arr)


        ax.plot(f_arr_GHz, 20 * np.log10(np.abs(self.Smtx_j0[0, 0])),
                # label=label_prepend + '\nJ=' + str(np.round(self.jpa_val, 4)),
                linestyle=linestyle,
                color=primary_color
                )
        ax.legend()
        return fig

    def total_passive_ABCD_func(self, omega_s, omega_i, add_index = 0):
        '''
        Let's calculate the scattering parameters of the network when the JPA is
        off. This should be easy, we're just looking at the phase structure of the
        network, the ABCD matrices are all multiplied together, and then transformed
        to scattering matrices
        '''
        # find out where the inverter is
        inverter_index = \
        [(i, el) for (i, el) in enumerate(self.circuit_elements) if type(el) == CoreResonator][0][0]

        print("Generating ABCD Matrices...")
        self.ABCD_mtxs_vs_frequency = [abcd(omega_s, omega_i) for abcd in self.ABCD_methods[:inverter_index+add_index]]

        return compress_abcd_numerical(self.ABCD_mtxs_vs_frequency)  # omega_s just for length

    def passive_impedance_seen_from_port(self, add_index=0):
        '''
        This function calculates the impedance seen from the outside port.
        This is just Z00 for a one port network
        '''
        ABCD = self.total_passive_ABCD(array=True, add_index = add_index)
        Z = abcd_to_z(ABCD, self.Z0)
        return Z[0,0]

    def passive_impedance_seen_from_core_mode(self, add_index=0, debug = False):
        '''
        This function calculates the impedance seen from the array port
        of the network, without including the array mode at all.
        '''

        ABCD = self.total_passive_ABCD(array=False, add_index = -1+add_index, debug = debug)
        if self.Ftype == 'cap_cpld_lumped' or self.Ftype == 'cap_cpld_l4':
            negative_first_cap_symbol = sp.symbols('C_comp')
            negative_first_cap = Capacitor(omega_symbol=self.omega_from_inverter, symbol = negative_first_cap_symbol, val = -self.CC[0])
            ABCD_comp = negative_first_cap.ABCDshunt()
            self.net_subs.append((negative_first_cap_symbol, negative_first_cap.val))
            ABCD_total = ABCD * ABCD_comp
        else:
            ABCD_total = ABCD
        Z = abcd_to_z(ABCD_total, self.Z0)
        return Z[1, 1] - Z[0, 1] * Z[1, 0] / (Z[0, 0] + self.Z0)

    def passive_impedance_seen_from_array(self, add_index = 0):
        '''
        This function calculates the impedance seen from the array port
        of the network including the array inductance
        '''
        ABCD = self.total_passive_ABCD(array=False, add_index = add_index)

        Z = abcd_to_z(ABCD, self.Z0)

        return Z[1, 1] - Z[0, 1] * Z[1, 0] / (Z[0, 0] + self.Z0)

    def passive_impedance_seen_from_inverter(self, add_index = 0, debug = False):
        '''
        This function calculates the impedance seen from the array port
        of the network including the array inductance
        '''
        ABCD = self.total_passive_ABCD(array=True, add_index = add_index, debug = debug)

        Z = abcd_to_z(ABCD, self.Z0)

        return Z[1, 1] - Z[0, 1] * Z[1, 0] / (Z[0, 0] + self.Z0)
