from parametricSynthesis.network_tools.network_synthesis import (get_active_network_prototypes,
                                                                 get_passive_network_prototypes,
                                                                 calculate_network)
from parametricSynthesis.network_tools.prototype_calc import prototype_gs
import numpy as np
import matplotlib.pyplot as plt

from parametricSynthesis.network_tools.network_synthesis import (get_active_network_prototypes,
                                                                 get_passive_network_prototypes,
                                                                 calculate_network)
from parametricSynthesis.network_tools.prototype_calc import prototype_gs
import numpy as np
import matplotlib.pyplot as plt

net_dict = {}
active_network_prototypes = get_active_network_prototypes()
passive_network_prototypes = get_passive_network_prototypes()
f0 = 7e9
w0 = 2 * np.pi * f0
dw = 0.05
# Z_squid = 25
L_squid = 0.5e-9
# L_squid = w0*Z_squid
Z_squid = w0 * L_squid

# g_arr = np.array([1, 1.5963, 1.0967, 1.5963, 1])
#^^^pozar example
g_arr_old = active_network_prototypes['N3_Cheby_20dB_R05']
g_arr = prototype_gs(20, type = 'chebyshev', n = 3, r_db = 2)
z_arr = np.array([0, 35*4/np.pi, 35*4/np.pi, 50], dtype=float)
tline_corr_factor = 1.005

f_arr_GHz = np.linspace(f0 / 1e9 - 0.5, f0 / 1e9 + 0.5, 401)
net = calculate_network(g_arr, z_arr, f0, dw, L_squid, printout=False, inv_corr_factor=tline_corr_factor)

net.gen_net_by_type('cap_cpld_l4', active = True, core_inductor = False, method = 'pumped_mutual',
                      tline_inv_Z_corr_factor = 1, use_approx = False) #0.945
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


del_fig, del_ax = plt.subplots()
omega_arr = net.omega_plot_arr
del_arr = -np.gradient(np.unwrap(np.arctan2(net.Smtx_j0[0,0,:].imag, net.Smtx_j0[0,0,:].real)))/np.gradient(omega_arr)*1e9
del_ax.plot(omega_arr/2/np.pi, del_arr)
del_ax.grid()

# plt.show()
breakpoint()