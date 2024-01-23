from parametricSynthesis import network_tools, drawing_tools, simulation_tools
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import schemdraw.elements as elm
import schemdraw
import plotly.express as px
import pandas as pd

f0 = 7e9
w0 = 2 * np.pi * f0
dw = 0.25
L_squid = 0.7e-9
# g_arr = active_network_prototypes['N3_Cheby_20dB_R05']
# g_arr = [1.0, 0.6545, 0.7488, 0.4397, 0.9045] #cheby with 1dB ripple
# g_arr = [1.0, 0.7020, 0.6886, 0.4060, 0.8674]#17db leg with 1dB ripple
# g_arr = [1.0, 0.5849, 0.6527, 0.3534, 0.9045] #leg with 1dB ripple
g_arr = [1.0, 0.4084, 0.4399, 0.2250, 0.9045]  # leg with 0.1dB ripple
# g_arr = [1.0, 0.5244, 0.5778, 0.3055, 0.9045] #leg with 0.5dB ripple

# g_arr = [1.0, 0.9598, 1.1333, 1.3121, 0.4440, 1.1528] #17dB leg filter with 4 nodes
# g_arr = [1.0, 0.5833, 0.7364, 0.7160, 0.2244, 1.1055] #20dB leg filter with 4 nodes, 0.1dB ripple
z_arr = np.array([0, 50, 0, 50], dtype=float)
f_arr_GHz = np.linspace(f0 / 1e9 - 0.7, f0 / 1e9 + 0.7, 201)
net = network_tools.network_synthesis.calculate_network(g_arr, z_arr, f0, dw, L_squid, printout=False)
# net_types = ['tline_cpld_lumped', 'tline_cpld_l2', 'tline_cpld_l4']
# linestyles = ['dotted', 'dashed', 'solid']
net_types = ['tline_cpld_l4']
linestyles = ['solid']
fig, ax = plt.subplots()
ax.set_title('Gain by pump power and filter type')
ax.grid()

# pumped mutual
for net_type, linestyle in zip(net_types, linestyles):
    net.gen_net_by_type(net_type, active=True, core_inductor=False, method='pumped_mutual',
                        tline_inv_Z_corr_factor=0.95, use_approx=False)  # 0.945

    fig = net.plot_scattering(f_arr_GHz,
                              linestyle=linestyle,
                              fig=fig,
                              vary_pump=False,
                              method='pumped_mutual',
                              focus=True,
                              primary_color='b',
                              label_prepend='pumped mutual ',
                              debug=True);
ax.legend(bbox_to_anchor=(1, 1), ncol=3)
plt.show(fig)