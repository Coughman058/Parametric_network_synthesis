from parametricSynthesis.network_tools.network_synthesis import (get_active_network_prototypes,
                                                                 get_passive_network_prototypes,
                                                                 calculate_network)
import numpy as np
import matplotlib.pyplot as plt

net_dict = {}
active_network_prototypes = get_active_network_prototypes()
passive_network_prototypes = get_passive_network_prototypes()
f0 = 7e9
w0 = 2 * np.pi * f0
dw = 0.1
# Z_squid = 25
L_squid = 1e-9
Z_squid = w0 * L_squid

# g_arr = np.array([1, 1.5963, 1.0967, 1.5963, 1])
#^^^pozar example
g_arr = active_network_prototypes['N2_Cheby_20dB_R05']
z_arr = np.array([0, 50, 50], dtype=float)
tline_corr_factor = 0.9

f_arr_GHz = np.linspace(f0 / 1e9 - 0.5, f0 / 1e9 + 0.5, 401)
net = calculate_network(g_arr, z_arr, f0, dw, L_squid, printout=False)

net.gen_net_by_type('tline_cpld_l4', active = True, core_inductor = False, method = 'pumped_mutual',
                      tline_inv_Z_corr_factor = tline_corr_factor, use_approx = False) #0.945
fig, ax = plt.subplots()
ax.set_title('Gain by pump power and filter type')
ax.grid()

fig = net.plot_scattering(f_arr_GHz,
                        linestyle = 'solid',
                        fig = fig,
                        vary_pump = False,
                        method = 'pumped_mutual',
                        focus = True,
                        primary_color = 'b',
                        label_prepend='pumped mutual ',
                        debug = True);
# plt.show()
breakpoint()