from parametricSynthesis.network_tools.network_synthesis import (get_active_network_prototypes,
                                                                 get_passive_network_prototypes,
                                                                 calculate_network)
import numpy as np

net_dict = {}
active_network_prototypes = get_active_network_prototypes()
passive_network_prototypes = get_passive_network_prototypes()
f0 = 7e9
w0 = 2 * np.pi * f0
dw = 0.1
L_squid = 0.5e-9
# Z_squid = w0 * L_squid
Z_squid = 50
g_arr = passive_network_prototypes['N3_Cheby_R05']
z_arr = np.array([50, 50, 50], dtype=float)
tline_corr_factor = 0.85
f_arr_GHz = np.linspace(f0 / 1e9 - 0.7, f0 / 1e9 + 0.7, 201)
net = calculate_network(g_arr, z_arr, f0, dw, L_squid, printout=False)

breakpoint()