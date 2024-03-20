from dataclasses import dataclass
import sympy as sp
import numpy as np
from IPython.display import display
from typing import Union
from skrf.circuit import Circuit
from skrf.network import Network
from tqdm import tqdm
import schemdraw
import schemdraw.elements as elm
'''
we need classes for each circuit element, in particular, functions that can
leverage sympy to give symbolic scattering information would be very helpful
'''

def z_to_input_impedance(z: Union[sp.Matrix, np.array], Z0: Union[sp.Symbol, float], num=False):
    Z11 = z[0, 0]
    Z12 = z[0, 1]
    Z21 = z[1, 0]
    Z22 = z[1, 1]
    Zin = (Z11 + Z12 / Z0 + Z21 * Z0 + Z22)
    return Zin

def abcd_to_s(abcd: Union[sp.Matrix, np.array], Z0: Union[sp.Symbol, float], num=True):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    denom = (A + B / Z0 + C * Z0 + D)

    if num:
        Smtx = np.array(
            [[(A + B / Z0 - C * Z0 - D) / denom, 2 * (A * D - B * C) / denom],
             [2 / denom, (-A + B / Z0 + C * Z0 + D) / denom]]
        )
    else:
        Smtx = sp.Matrix(
            [[(A + B / Z0 - C * Z0 - D) / denom, 2 * (A * D - B * C) / denom],
             [2 / denom, (-A + B / Z0 + C * Z0 + D) / denom]]
        )
    return Smtx


def abcd_to_z(abcd: Union[sp.Matrix, np.array], num=True):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]

    if num:
        Zmtx = np.array(
            [[A / C, (A * D - B * C) / C],
             [1 / C, D / C]]
        )
    else:
        Zmtx = sp.Matrix(
            [[A / C, (A * D - B * C) / C],
             [1 / C, D / C]]
        )

    return Zmtx


def compress_abcd_array(ABCD_mtxs: list, simplify=False, mid_simplify_rules=[], debug=False, ):
    total_ABCD = sp.eye(2)
    for i, ABCD in enumerate(ABCD_mtxs):
        total_ABCD *= ABCD
        if simplify:
            total_ABCD = sp.simplify(total_ABCD.subs(mid_simplify_rules))
        if debug:
            display(total_ABCD)
    return total_ABCD
def compress_abcd_numerical(ABCD_list: list):
    total_ABCD = np.tile(np.eye(2), (ABCD_list[0].shape[0],1,1)) #ABCD matrix at every index
    print("compressing ABCD matrices...")
    for i, el_ABCD in enumerate(tqdm(ABCD_list)):
        total_ABCD = np.matmul(total_ABCD, el_ABCD)
    return total_ABCD

def cap_to_tline_length_mod_factor(cap,
                                   z0,
                                   omega0):
    length_factor = (1-z0*omega0*cap*2/np.pi)
    print("compensating z = ", z0, "tline with cap = ", cap, "length factor = ", length_factor)
    return length_factor

@dataclass
class Resistor:
    # omega_symbol:sp.Symbol
    symbol: sp.Symbol
    val: float

    def impedance_symbolic(self):
        return self.symbol

    def impedance_function(self, omega):
        return self.val

    def ABCDseries(self):
        return sp.Matrix([[1, self.impedance_symbolic()], [0, 1]])

    def ABCDshunt(self):
        return sp.Matrix([[1, 0], [1 / self.impedance_symbolic(), 1]])


@dataclass
class Capacitor:
    omega_symbol: sp.Symbol
    symbol: sp.Symbol
    val: float

    def impedance_symbolic(self):
        return 1 / (sp.I * self.symbol * self.omega_symbol)

    def impedance_function(self, omega):
        return sp.lambdify(self.omega_symbol, self.impedance_symbolic().subs(self.symbol, self.val))(omega)

    def ABCDseries(self):
        return sp.Matrix([[1, self.impedance_symbolic()], [0, 1]])

    def ABCDshunt(self):
        return sp.Matrix([[1, 0], [1 / self.impedance_symbolic(), 1]])

    def ABCDseries_function(self, omega):
        return np.moveaxis(np.array([[1*np.ones_like(omega), self.impedance_function(omega)],
                                     [0*np.ones_like(omega), 1*np.ones_like(omega)]]),-1,0)

    def ABCDshunt_function(self, omega):
        return np.moveaxis(np.array([[1*np.ones_like(omega), 0*np.ones_like(omega)],
                                     [1 / self.impedance_function(omega), 1*np.ones_like(omega)]]),-1,0)


@dataclass
class Inductor:
    omega_symbol: sp.Symbol
    symbol: sp.Symbol
    val: float

    def impedance_symbolic(self):
        return (sp.I * self.symbol * self.omega_symbol)

    def impedance_function(self, omega):
        return sp.lambdify(self.omega_symbol, self.impedance_symbolic().subs(self.symbol, self.val))(omega)

    def ABCDseries(self):
        return sp.Matrix([[1, self.impedance_symbolic()], [0, 1]])

    def ABCDshunt(self):
        return sp.Matrix([[1, 0], [1 / self.impedance_symbolic(), 1]])

    def ABCDseries_function(self, omega):
        return np.array([[1*np.ones_like(omega), self.impedance_function(omega)], [0*np.ones_like(omega), 1*np.ones_like(omega)]])

    def ABCDshunt_function(self, omega):
        return np.moveaxis(np.array([[1*np.ones_like(omega), 0*np.ones_like(omega)],
                         [1 / self.impedance_function(omega), 1*np.ones_like(omega)]]), -1,0)


@dataclass
class Tline_short:
    omega_symbol: sp.Symbol
    Z_symbol: sp.Symbol
    theta_symbol: sp.Symbol
    omega0_symbol: sp.Symbol

    Zval: float
    theta_val: float
    omega0_val: float

    def short_impedance_symbolic(self, use_approx = False):
        z0 = self.Z_symbol
        bl = self.theta_symbol * self.omega_symbol / self.omega0_symbol

        if use_approx == False:
            cotan = sp.cot(bl)
        else:
            cotan = sp.series(sp.cot(bl), self.omega_symbol, x0=self.omega0_symbol, n=4).removeO()

        return 1/(-sp.I * cotan / z0)
    def open_impedance_symbolic(self, use_approx = False):
        z0 = self.Z_symbol
        bl = self.theta_symbol * self.omega_symbol / self.omega0_symbol

        if use_approx == False:
            tan = sp.tan(bl)
        else:
            tan = sp.series(sp.tan(bl), self.omega_symbol, x0=self.omega0_symbol, n=4).removeO()

        return 1/(sp.I * tan / z0)
    def impedance_function(self, use_approx = False):
        return sp.lambdify(self.omega_symbol,
                           self.short_impedance_symbolic(use_approx=use_approx).subs([
                               (self.Z_symbol, self.Zval),
                               (self.theta_symbol, self.theta_val),
                               (self.omega0_symbol, self.omega0_val)]))

    def ABCDshunt(self, use_approx=False):
        return np.array([[1, 0], [1/self.short_impedance_symbolic(use_approx=use_approx), 1]])

    def ABCDshunt_function(self, omega_arr):
        return np.moveaxis(np.array([[1*np.ones_like(omega_arr), 0*np.ones_like(omega_arr)],
                          [1/self.impedance_function()(omega_arr), 1*np.ones_like(omega_arr)]]),
                           -1,0)


@dataclass
class Tline_open:
    omega_symbol: sp.Symbol
    Z_symbol: sp.Symbol
    theta_symbol: sp.Symbol
    omega0_symbol: sp.Symbol

    Zval: float
    theta_val: float
    omega0_val: float

    def impedance_symbolic(self, use_approx = False):
        z0 = self.Z_symbol
        bl = self.theta_symbol * self.omega_symbol / self.omega0_symbol

        if use_approx == False:
            tan = sp.tan(bl)
        else:
            tan = sp.series(sp.tan(bl), self.omega_symbol, x0=self.omega0_symbol, n=4).removeO()

        return 1/(sp.I * tan / z0)
    def impedance_function(self, use_approx = False):
        return sp.lambdify(self.omega_symbol,
                           self.impedance_symbolic(use_approx=use_approx).subs([
                                                                                (self.Z_symbol, self.Zval),
                                                                                (self.theta_symbol, self.theta_val),
                                                                                (self.omega0_symbol, self.omega0_val)])
                           )


    def ABCDseries(self, use_approx=False):
        z0 = self.Z_symbol
        bl = self.theta_symbol * self.omega_symbol / self.omega0_symbol
        if use_approx == False:
            cos = sp.cos(bl)
            sin = sp.sin(bl)
        else:
            cos = sp.series(sp.cos(bl), self.omega_symbol, x0=self.omega0_symbol, n=4).removeO()
            sin = sp.series(sp.sin(bl), self.omega_symbol, x0=self.omega0_symbol, n=4).removeO()

        return sp.Matrix([[cos, sp.I * z0 * sin], [sp.I * 1 / z0 * sin, cos]])

    def ABCDshunt_terminated(self, Zl, use_approx=False):
        z0 = self.Z_symbol
        bl = self.theta_symbol * self.omega_symbol / self.omega0_symbol

        if use_approx == False:
            tan = sp.tan(bl)
        else:
            tan = sp.series(sp.tan(bl), self.omega_symbol, x0=self.omega0_symbol, n=4).removeO()

        return sp.Matrix([[1, 0], [1 / z0 * ((z0 + sp.I * Zl * tan) / (Zl + sp.I * z0 * tan)), 1]])

    def ABCDshunt(self, use_approx=False):

        return sp.Matrix([[1, 0], [1/self.impedance_symbolic(use_approx=use_approx), 1]])

    def ABCDshunt_function(self, omega_arr):
        return np.moveaxis(np.array([[1*np.ones_like(omega_arr), 0*np.ones_like(omega_arr)],
                          [1/self.impedance_function(omega_arr), 1*np.ones_like(omega_arr)]]), -1,0)

    def ABCDseries_function(self, omega_arr, use_approx = False):
        return np.moveaxis(sp.lambdify(self.omega_symbol,
                           self.ABCDseries(use_approx = use_approx).subs([
                               (self.Z_symbol, self.Zval),
                               (self.theta_symbol, self.theta_val),
                               (self.omega0_symbol, self.omega0_val)]))(omega_arr), -1,0)

@dataclass
class DegenerateParametricInverterAmp:
    omega0_val: float
    omega1: sp.Symbol
    omega2: sp.Symbol
    L: sp.Symbol
    Lval: float
    R_active: sp.Symbol
    Jpa_sym: sp.Symbol
    power_G_db: float
    g_arr: list
    net_size: int
    dw: float

    def __post_init__(self):
        self.Zcore = self.omega0_val * self.Lval
        # print(f"Network INGREDIENTS: dw = {self.dw}\nZcore = {self.Zcore}\ng_arr = {self.g_arr}\ng0 = {self.g_arr[0]}\ng1 = {self.g_arr[1]}\ngN+1 = {self.g_arr[-1]}")

        alpha = self.alpha_val_from_matching_params(10 ** self.power_G_db / 10, self.dw, self.g_arr)
        Lprime = self.Lval
        print("ALPHA = ", alpha)
        # Jpa_val_s = 1 / omega_s_arr / Lprime / (1 - alpha) * np.sqrt(alpha)
        # Jpa_val_i = 1 / np.abs(omega_i_arr) / Lprime / (1 - alpha) * np.sqrt(alpha)

        if np.size(self.g_arr) % 2 == 0:
            Jpa_test = self.dw / self.Zcore / self.g_arr[1] / np.sqrt(self.g_arr[0]) * np.sqrt(self.g_arr[-1])
        else:
            Jpa_test = self.dw / self.Zcore / self.g_arr[1] / np.sqrt(self.g_arr[0]) / np.sqrt(self.g_arr[-1])
        # Jpa_test = self.dw / self.Zcore / self.g_arr[1] / np.sqrt(self.g_arr[0]) * np.sqrt(self.g_arr[-1])
        self.Jpa_val_s = Jpa_test
        self.Jpa_val_i = Jpa_test

        self.signal_inductor = Inductor(self.omega1, self.L, Lprime)
        self.idler_inductor = Inductor(self.omega2, self.L, Lprime)

    def abcd_inverter_shunt(self, omega_s_arr, omega_i_arr):



        inv_ABCD = np.array([[0*np.ones_like(omega_s_arr), 1j / np.conjugate(self.Jpa_val_s)*np.ones_like(omega_s_arr)],
                             [-1j * self.Jpa_val_i*np.ones_like(omega_s_arr), 0*np.ones_like(omega_s_arr)]])
        return np.moveaxis(inv_ABCD, -1,0)

    # def dm(self):
    #     return 2 * sp.sqrt(self.L * self.L * self.alpha)

    def alpha_val_from_matching_params(self, G, frac_bw, g_arr):
        if np.size(g_arr) % 2 == 0:
            return (np.sqrt(G) - 1) / (np.sqrt(G) + 1) * (frac_bw / g_arr[1]) ** 2 * g_arr[-1] ** 2
        else:
            return (np.sqrt(G) - 1) / (np.sqrt(G) + 1) * (frac_bw / g_arr[1]) ** 2 / g_arr[-1] ** 2

    def abcd_function(self, omega_s_arr, omega_i_arr):
        A1 = self.signal_inductor.ABCDshunt_function(omega_s_arr)
        A2 = self.abcd_inverter_shunt(omega_s_arr, omega_i_arr)
        A3 = self.idler_inductor.ABCDshunt_function(omega_i_arr)
        return A1 @ A2 @ A3

@dataclass
class Coupler:
    """
    class containing all parameters of any given coupler in the filter network, whether it's lumped, tline, etc.
    """
    n: int
    omega_sym: sp.Symbol
    omega0_sym: sp.Symbol
    omega0_val: float
    j_val: float
    cap_comp_val: float = 0
    ind_comp_val: float = 0
    signal_or_idler_flag: str = 'signal'
    def abcd_function(self, omega_s, omega_i):
        pass

@dataclass
class Resonator:
    """
    class containing all parameters of any given resonator in the filter network, whether it's lumped, tline, etc.
    All the functions except the compensation function need to be overridden as they are placeholders for the subclasses
    """
    n: int
    omega_sym: sp.Symbol
    omega0_sym: sp.Symbol
    omega0_val: float
    z_sym: sp.Symbol
    z_val: float
    signal_or_idler_flag: str

    def abcd_function(self, omega_s, omega_i):
        '''
        Function that returns the ABCD matrix as a function of frequency for a shunt resonator
        in the format Nx(2x2) where N is the number of frequency points
        '''
        pass

    def compensate_for_couplers(self, c1: Coupler, c2: Coupler):
        '''
        Function that compensates the resonator for the coupling capacitors according to their comp_cap_val
        '''
        pass

@dataclass
class LumpedResonator(Resonator):

    def __post_init__(self):
        """
        Adds a lumped resonator to the network
        :param n: location of the resonator in the network, this will be used to name the elements and assign the values
        to variables
        :param net_size: number of resonators in the network. This is actually unused in this function, but is used in
        the lumped_res_compensated function
        :param omega_sym: the symbol for the angular frequency
        :param include_inductor: whether to include an inductor in the resonator. This is used for the first resonator
        in the network, which has the parametric inverter element that includes the inductor
        :param compensated: whether the capacitors in the resonator are compensated by coupling caps or not
        :return:
        """
        self.ind_sym = sp.symbols("L" + str(self.n))
        self.cap_sym = sp.symbols("C" + str(self.n))
        self.ind_val = self.z_val / self.omega0_val
        self.cap_val = 1 / (self.z_val * self.omega0_val)

    def compensate_for_couplers(self, c1, c2):
        print("compensating resonator ", self.n, "with caps ", c1.cap_comp_val, "and ", c2.cap_comp_val)
        self.cap_val -= (c1.cap_comp_val+c2.cap_comp_val)

    def synthesize(self):
        self.ind_el = Inductor(self.omega_sym, self.ind_sym, self.ind_val)
        self.cap_el = Capacitor(self.omega_sym, self.cap_sym, self.cap_val)

    def abcd_function(self, omega_s_arr, omega_i_arr):
        if self.signal_or_idler_flag == 'signal':
            omega = omega_s_arr
        else:
            omega = omega_i_arr
        return self.ind_el.ABCDshunt_function(omega) @ self.cap_el.ABCDshunt_function(omega)

    def add_to_drawing(self, d, l = 1.5):
        '''
        Adds the resonator to a drawing object from schemdraw going right from the port
        '''
        d += elm.Inductor2().down().label(f"$L =$ {np.round(self.ind_val * 1e12, 1)} pH", loc='bottom')
        d += elm.Line().right().length(l / 2)
        d.pop()
        d += elm.Line().right().length(l)
        d.push()
        d += elm.Capacitor().down().label(f"\n\n$C =$ {np.round(self.cap_val * 1e12, 3)} pF", loc='top')
        d += elm.Line().left().length(l / 2)
        d += elm.Ground()
        d.pop()

@dataclass
class CoreResonator:
    n: int
    omega_s_sym: sp.Symbol
    omega_i_sym: sp.Symbol
    omega0_sym: sp.Symbol
    omega0_val: float
    z_sym: sp.Symbol
    z_val: float
    jpa_sym: sp.Symbol
    power_G_db: float
    g_arr: list
    net_size: int
    dw: float

    def __post_init__(self):
        self.ind_sym = sp.symbols("L" + str(self.n))
        self.R_active_sym = sp.symbols("R" + str(self.n))
        self.cap_s_sym = sp.symbols("Cs" + str(self.n))
        self.cap_i_sym = sp.symbols("Ci" + str(self.n))
        self.ind_val = self.z_val / self.omega0_val
        # print("Inductor VALUE = ", self.ind_val)
        self.cap_val = 1 / (self.z_val * self.omega0_val)
        # print("Capacitor VALUE = ", self.cap_val)

    def compensate_for_couplers(self, c1):
        self.cap_val -= c1.cap_comp_val

    def synthesize(self):
        self.inv_el = DegenerateParametricInverterAmp(self.omega0_val,
                                                      self.omega_s_sym,
                                                      self.omega_i_sym,
                                                      self.ind_sym,
                                                      self.ind_val,
                                                      self.R_active_sym,
                                                      self.jpa_sym,
                                                      self.power_G_db,
                                                      self.g_arr,
                                                      self.net_size,
                                                      self.dw)

        self.cap_s_el = Capacitor(self.omega_s_sym, self.cap_s_sym, self.cap_val)
        self.cap_i_el = Capacitor(self.omega_i_sym, self.cap_i_sym, self.cap_val)

    def abcd_function(self, omega_s_arr, omega_i_arr):
        return (self.cap_s_el.ABCDshunt_function(omega_s_arr) @
                self.inv_el.abcd_function(omega_s_arr, omega_i_arr) @
                self.cap_i_el.ABCDshunt_function(omega_i_arr))

    def add_to_drawing(self, d, l = 1.5):
        '''
        Adds the resonator to a drawing object from schemdraw going right from the port
        '''
        d += elm.Inductor2().down().label(f"$L =$ {np.round(self.ind_val * 1e12, 1)} pH", loc='bottom')
        d += elm.Line().right().length(l / 2)
        d.pop()
        d += elm.Line().right().length(l)
        d.push()
        d += elm.Capacitor().down().label(f"\n\n$C =$ {np.round(self.cap_val * 1e12, 3)} pF", loc='top')
        d += elm.Line().left().length(l / 2)
        d += elm.Ground()
        d.pop()

@dataclass
class TlineL4Resonator(Resonator):

    def __post_init__(self):
        """
        Adds a lumped resonator to the network
        :param n: location of the resonator in the network, this will be used to name the elements and assign the values
        to variables
        :param net_size: number of resonators in the network. This is actually unused in this function, but is used in
        the lumped_res_compensated function
        :param omega_sym: the symbol for the angular frequency
        :param include_inductor: whether to include an inductor in the resonator. This is used for the first resonator
        in the network, which has the parametric inverter element that includes the inductor
        :param compensated: whether the capacitors in the resonator are compensated by coupling caps or not
        :return:
        """
        self.theta_sym = sp.symbols("theta_" + str(self.n))
        self.z_sym = sp.symbols("Zt_" + str(self.n))
        self.theta_val = np.pi/2
        self.z_tline = self.z_val*np.pi/4

    def compensate_for_couplers(self, c1, c2):
        self.theta_val *= cap_to_tline_length_mod_factor(c1.cap_comp_val+c2.cap_comp_val, self.z_tline, self.omega0_val)

    def synthesize(self):
        self.tline_el = Tline_short(self.omega_sym,
                                    self.z_sym,
                                    self.theta_sym,
                                    self.omega0_sym,
                                    self.z_tline,
                                    self.theta_val,
                                    self.omega0_val)

    def abcd_function(self, omega_s_arr, omega_i_arr):
        if self.signal_or_idler_flag == 'signal':
            omega = omega_s_arr
        else:
            omega = omega_i_arr
        return self.tline_el.ABCDshunt_function(omega)

@dataclass
class CapCoupler(Coupler):
    def __post_init__(self):
        self.cap_sym = sp.symbols("Cc_" + str(self.n))
        self.cap_val = 1/self.omega0_val*self.j_val
        self.cap_comp_val = self.cap_val

    def compensate_end_capacitor(self, zl):
        '''
        two things need to happen here:

        one: this capacitor's main value is divided by the sqrt of the mod factor,
        this happens in regular synthesis just because the bounds have no
        reactive part

        two: the compensation value is bumped *up* by the sqrt of the mod factor compared to the base capacitance,
        because we have to compensate the
        only adjacent resonator more because it is the *only* adjacent resonator

        zl is the load impedance (resistance) that we are terminating against
        '''

        mod_factor = (1 - zl ** 2 * self.j_val ** 2)
        self.cap_comp_val = self.cap_val * np.sqrt(mod_factor)
        self.cap_val = self.cap_val / np.sqrt(mod_factor)

    def synthesize(self):
        self.cap_el = Capacitor(self.omega_sym, self.cap_sym, self.cap_val)

    def abcd_function(self, omega_s_arr, omega_i_arr):
        if self.signal_or_idler_flag == 'signal':
            omega = omega_s_arr
        else:
            omega = omega_i_arr
        return self.cap_el.ABCDseries_function(omega)

@dataclass
class TlineCoupler(Coupler):

    def __post_init__(self):
        self.theta_sym = sp.symbols("theta_c_" + str(self.n))
        self.z_sym = sp.symbols("Zt_c_" + str(self.n))
        self.theta_val = sp.pi/2
        self.z_tline = 1/self.j_val
        self.tline_el = Tline_open(self.omega_sym,
                                   self.z_sym,
                                   self.theta_sym,
                                   self.omega0_sym,
                                   self.z_tline,
                                   self.theta_val,
                                   self.omega0_val)
    def synthesize(self):
        pass

    def compensate_end_capacitor(self, zl):
        pass
    def abcd_function(self, omega_s_arr, omega_i_arr):
        if self.signal_or_idler_flag == 'signal':
            omega = omega_s_arr
        else:
            omega = omega_i_arr
        return self.tline_el.ABCDseries_function(omega)

    def add_to_drawing(self, d , l = 1.5):
        d += elm.Coax(
            label=f"$\lambda/4$\n$Z_c$ = {np.round(1 / self.z_tline, 1)} $\Omega$").scale(
            1).right()
        d.push()


