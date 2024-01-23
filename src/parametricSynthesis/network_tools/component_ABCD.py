from dataclasses import dataclass
import sympy as sp
import numpy as np
from IPython.display import display
from typing import Union
'''
we need classes for each circuit element, in particular, functions that can
leverage sympy to give symbolic scattering information would be very helpful
'''

def ABCD_to_S(abcd: Union[sp.Matrix, np.array], Z0: Union[sp.Symbol, float], num = False):
  A = abcd[0,0]
  B = abcd[0,1]
  C = abcd[1,0]
  D = abcd[1,1]
  denom = (A+B/Z0+C*Z0+D)

  if num:
    Smtx = np.array(
      [[(A+B/Z0-C*Z0-D)/denom,2*(A*D-B*C)/denom],
       [2/denom,(-A+B/Z0+C*Z0+D)/denom]]
      )
  else:
    Smtx = sp.Matrix(
      [[(A+B/Z0-C*Z0-D)/denom,2*(A*D-B*C)/denom],
       [2/denom,(-A+B/Z0+C*Z0+D)/denom]]
      )

  return Smtx

def ABCD_to_Z(abcd: Union[sp.Matrix, np.array], num = False):
  A = abcd[0,0]
  B = abcd[0,1]
  C = abcd[1,0]
  D = abcd[1,1]

  if num:
    Zmtx = np.array(
      [[A / C, (A * D - B * C) / C],
       [1 / C, D / C]]
      )
  else:
    Smtx = sp.Matrix(
      [[A / C, (A * D - B * C) / C],
       [1 / C, D / C]]
      )

  return Zmtx


def compress_ABCD_array(ABCD_mtxs: list, simplify = False, mid_simplify_rules = [], debug = False, ):
  total_ABCD = sp.eye(2)
  for i, ABCD in enumerate(ABCD_mtxs):
    total_ABCD *= ABCD
    if simplify:
      total_ABCD = sp.simplify(total_ABCD.subs(mid_simplify_rules))
    if debug:
      display(total_ABCD)
  return total_ABCD

@dataclass
class resistor:
  # omega_symbol:sp.Symbol
  symbol:sp.Symbol
  val:float

  def impedance_symbolic(self):
    return self.symbol
  def impedance_function(self, omega):
    return self.val
  def ABCDseries(self):
    return sp.Matrix([[1,self.impedance_symbolic()],[0,1]])
  def ABCDshunt(self):
    return sp.Matrix([[1,0],[1/self.impedance_symbolic(),1]])

@dataclass
class capacitor:
  omega_symbol:sp.Symbol
  symbol:sp.Symbol
  val:float

  def impedance_symbolic(self):
    return 1/(sp.I*self.symbol*self.omega_symbol)
  def impedance_function(self, omega):
    return sp.lambdify(omega, self.impedance_symbolic().subs(self.symbol, self.val))
  def ABCDseries(self):
    return sp.Matrix([[1,self.impedance_symbolic()],[0,1]])
  def ABCDshunt(self):
    return sp.Matrix([[1,0],[1/self.impedance_symbolic(),1]])

@dataclass
class inductor:
  omega_symbol:sp.Symbol
  symbol:sp.Symbol
  val:float

  def impedance_symbolic(self):
    return (sp.I*self.symbol*self.omega_symbol)
  def impedance_function(self, omega):
    return sp.lambdify(omega, self.impedance_symbolic().subs(self.symbol, self.val))
  def ABCDseries(self):
    return sp.Matrix([[1,self.impedance_symbolic()],[0,1]])
  def ABCDshunt(self):
    return sp.Matrix([[1,0],[1/self.impedance_symbolic(),1]])

@dataclass
class TLINE:
  omega_symbol: sp.Symbol
  Z_symbol:sp.Symbol
  theta_symbol:sp.Symbol
  omega0_symbol: sp.Symbol

  Zval:float
  theta_val:float
  omega0_val:float

  def ABCDseries(self, use_approx = False):
    z0 = self.Z_symbol
    bl = self.theta_symbol*self.omega_symbol/self.omega0_symbol
    if use_approx == False:
      cos = sp.cos(bl)
      sin = sp.sin(bl)
    else:
      cos = sp.series(sp.cos(bl), self.omega_symbol, x0 = self.omega0_symbol, n = 4).removeO()
      sin = sp.series(sp.sin(bl), self.omega_symbol, x0 = self.omega0_symbol, n = 4).removeO()

    return sp.Matrix([[cos, sp.I*z0*sin],[sp.I*1/z0*sin,cos]])

  def ABCDshunt_terminated(self, Zl, use_approx = False):
    z0 = self.Z_symbol
    bl = self.theta_symbol*self.omega_symbol/self.omega0_symbol

    if use_approx == False:
      tan = sp.tan(bl)
    else:
      tan = sp.series(sp.tan(bl), self.omega_symbol, x0 = self.omega0_symbol, n = 4).removeO()

    return sp.Matrix([[1, 0],[1/z0*((z0+sp.I*Zl*tan)/(Zl+sp.I*z0*tan)), 1]])

  def ABCDshunt_open(self, use_approx = False):
    z0 = self.Z_symbol
    bl = self.theta_symbol*self.omega_symbol/self.omega0_symbol

    if use_approx == False:
      tan = sp.tan(bl)
    else:
      tan = sp.series(sp.tan(bl), self.omega_symbol, x0 = self.omega0_symbol, n = 4).removeO()

    return sp.Matrix([[1, 0],[sp.I*tan/z0, 1]])

  def ABCDshunt_short(self, use_approx = False):
    z0 = self.Z_symbol
    bl = self.theta_symbol*self.omega_symbol/self.omega0_symbol

    if use_approx == False:
      cotan = sp.cot(bl)
    else:
      cotan = sp.series(sp.cot(bl), self.omega_symbol, x0 = self.omega0_symbol, n = 4).removeO()

    return sp.Matrix([[1, 0],[-sp.I*cotan/z0, 1]])

@dataclass
class DegenerateParametricInverter_Amp:
  omega0_val: float
  omega1: sp.Symbol
  omega2: sp.Symbol
  L: sp.Symbol
  R_active: sp.Symbol
  Jpa_sym: sp.Symbol

  def __post_init__(self):
    self.Zcore = self.omega0_val*self.L
    # print(f"Network INGREDIENTS: dw = {self.dw}\nZcore = {self.Zcore}\ng_arr = {self.g_arr}\ng0 = {self.g_arr[0]}\ng1 = {self.g_arr[1]}\ngN+1 = {self.g_arr[-1]}")
    self.signal_inductor = inductor(self.omega1, self.L, self.L)
    self.idler_inductor = inductor(self.omega2, self.L, self.L)

  # def ABCD_series(self):
  #   '''
  #   This has to include both the series inductors and the inversion matrix
  #   '''
  #   inv_ABCD = sp.Matrix([[0, -sp.I*self.K1],[sp.I/sp.conjugate(self.K2), 0]])
  #   return self.signal_inductor.ABCDseries()*inv_ABCD*self.idler_inductor.ABCDseries()

  def ABCD_signal_inductor_shunt(self):
    return self.signal_inductor.ABCDshunt()

  def ABCD_inverter_shunt(self):

    inv_ABCD = sp.Matrix([[0, sp.I/self.Jpa_sym],[-sp.I*self.Jpa_sym, 0]])
    return inv_ABCD

  def ABCD_idler_inductor_shunt(self):
    return self.idler_inductor.ABCDshunt()

  def ABCD_pumpistor(self):
    inv_ABCD = sp.Matrix([[1,0],[1/self.R_active, 0]])
    return self.signal_inductor.ABCDshunt()*inv_ABCD

  def dM(self):
    return 2*sp.sqrt(self.L*self.L*self.alpha)

  def alpha_val_from_matching_params(self, G, frac_bw, g_arr):
    if np.size(g_arr)%2 == 0:
      return (np.sqrt(G)-1)/(np.sqrt(G)+1)*(frac_bw/g_arr[1])**2*g_arr[-1]**2
    else:
      return (np.sqrt(G)-1)/(np.sqrt(G)+1)*(frac_bw/g_arr[1])**2/g_arr[-1]**2