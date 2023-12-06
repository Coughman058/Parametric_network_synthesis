import sympy as sp
from copy import deepcopy
from .component_ABCD import inductor
import matplotlib.pyplot as plt
import numpy as np

def input_inductance_variation(net, L_wb_value, f_arr_Ghz, method = 'pumpistor'):
  L_wb_symbol = sp.symbols("L_{wb}")
  wirebond_inductor_signal = inductor(net.signal_omega_sym, L_wb_symbol, L_wb_value)
  wirebond_inductor_idler = inductor(net.idler_omega_sym, L_wb_symbol, L_wb_value)
  net_with_inductor = deepcopy(net)
  net_with_inductor.ABCD_mtxs.insert(0, wirebond_inductor_signal.ABCDseries())
  if method == 'pumped_mutual':
      net_with_inductor.ABCD_mtxs.append(wirebond_inductor_idler.ABCDseries())

  net_with_inductor.net_subs.append((L_wb_symbol, L_wb_value))

  #plot the difference
  fig, ax = plt.subplots()
  ax.set_title(f"Scattering vs. wirebond inductance: {np.round(L_wb_value*1e12, 0)}pH")
  fig = net.plot_scattering(f_arr_Ghz,
                            linestyle = 'solid',
                            fig = fig,
                            label_prepend = 'no wb\n',
                            focus=True,
                            vary_pump=False,
                            method = method
                            );
  fig = net_with_inductor.plot_scattering(f_arr_Ghz,
                                          linestyle = 'dashed',
                                          fig = fig,
                                          label_prepend = 'with wb\n',
                                          focus=True,
                                          vary_pump = False,
                                          method = method);
  ax.legend(ncol = 2, bbox_to_anchor = (1,1))
  ax.grid()
  return fig