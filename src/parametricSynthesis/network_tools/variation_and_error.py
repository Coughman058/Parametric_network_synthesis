import sympy as sp
from copy import deepcopy
from .component_ABCD import inductor
import matplotlib.pyplot as plt
import numpy as np

def input_inductance_variation(net, L_wb_value, f_arr_Ghz):
  L_wb_symbol = sp.symbols("L_{wb}")
  wirebond_inductor = inductor(net.signal_omega_sym, L_wb_symbol, L_wb_value)
  net_with_inductor = deepcopy(net)
  net_with_inductor.ABCD_mtxs.insert(0, wirebond_inductor.ABCDseries())
  net_with_inductor.net_subs.append((L_wb_symbol, L_wb_value))

  #plot the difference
  fig, ax = plt.subplots()
  ax.set_title(f"Scattering vs. wirebond inductance: {np.round(L_wb_value*1e12, 0)}pH")
  fig = net.plot_pumpistor_scattering(f_arr_Ghz,
                                      linestyle = 'solid',
                                      fig = fig,
                                      label_prepend = 'no wb\n');
  fig = net_with_inductor.plot_pumpistor_scattering(f_arr_Ghz,
                                      linestyle = 'dashed',
                                      fig = fig,
                                      label_prepend = 'with wb\n');
  ax.legend(ncol = 2, bbox_to_anchor = (1,1))
  ax.grid()
  return fig