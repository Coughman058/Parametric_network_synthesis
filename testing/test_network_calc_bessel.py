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
dw = 0.1
# Z_squid = 25
L_squid = 0.5e-9
# L_squid = w0*Z_squid
Z_squid = w0 * L_squid

# g_arr = np.array([1, 1.5963, 1.0967, 1.5963, 1])
#^^^pozar example
ftype = 'butterworth'
g_arr = prototype_gs(20, type = ftype, n = 3, r_db = 1)
z_arr = np.array([0, 50*4/np.pi, 50*4/np.pi, 50], dtype=float)
print("g_arr: ", g_arr)
# breakpoint()

tline_corr_factor = 0.99

f_arr_GHz = np.linspace(f0 / 1e9 - 0.5, f0 / 1e9 + 0.5, 401)
net = calculate_network(g_arr, z_arr, f0, dw, L_squid, printout=False, inv_corr_factor=tline_corr_factor)

net.gen_net_by_type('tline_cpld_l4', active = True, core_inductor = False, method = 'pumped_mutual',
                      tline_inv_Z_corr_factor = 1, use_approx = False) #0.945
fig, ax = plt.subplots()
ax.set_title(f'{ftype.capitalize()} Filter')
ax.grid()

fig = net.plot_scattering(f_arr_GHz,
                        linestyle = 'solid',
                        fig = fig,
                        vary_pump = False,
                        method = 'pumped_mutual',
                        focus = False,
                        primary_color = 'b',
                        label_prepend='pumped mutual ',
                        debug = True);

import sympy as sp
Lvary = sp.symbols("Lv")
passive_Y_func = sp.lambdify([net.omega_from_inverter, Lvary], 1/net.passive_impedance_seen_from_port().subs(net.inv_el.signal_inductor.symbol, Lvary).subs(net.net_subs))
passive_omega_arr = 2*np.pi*np.linspace(3e9, 11e9, 1001)
Lvary_val = L_squid*np.ones_like(passive_omega_arr)
#calculate passive scattering
S11 = (1/passive_Y_func(passive_omega_arr, Lvary_val)-50)/(1/passive_Y_func(passive_omega_arr, Lvary_val)+50)

del_fig, del_ax = plt.subplots()
del_ax.set_title("Group Delay (ns)")
omega_arr = net.omega_plot_arr
del_arr_active = -np.gradient(np.unwrap(np.arctan2(net.Smtx_j0[0,0,:].imag, net.Smtx_j0[0,0,:].real)))/np.gradient(omega_arr)*1e9
del_arr_passive = -np.gradient(np.unwrap(np.arctan2(S11.imag,S11.real)))/np.gradient(passive_omega_arr)*1e9
del_ax.plot(omega_arr/2/np.pi/1e9, del_arr_active)
del_ax.plot(passive_omega_arr/2/np.pi/1e9, del_arr_passive, label = f'passive, L = {L_squid*1e9} nH')
del_ax.grid()
del_ax.legend()
# plt.show()
breakpoint()