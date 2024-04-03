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
from parametricSynthesis.network_tools.component_ABCD import abcd_to_s

net_dict = {}
active_network_prototypes = get_active_network_prototypes()
passive_network_prototypes = get_passive_network_prototypes()
f0 = 6.5e9
w0 = 2 * np.pi * f0
dw = 0.15
# Z_squid = 25
L_squid = 0.37e-9
# L_squid = w0*Z_squid
Z_squid = w0 * L_squid

# g_arr = prototype_gs(20, type = 'chebyshev', n = 5, r_db = 1)
# z_arr = np.array([0, 20*4/np.pi, 30*4/np.pi, 30*4/np.pi, 20*4/np.pi, 50], dtype=float)
# resonator_types = ['core', 'l4', 'l4', 'l4', 'l4', 'l4', 'l4', 'l4', 'l4']
# coupler_types = ['cap', 'cap', 'cap', 'cap', 'cap', 'cap', 'cap', 'cap', 'cap']

power_G_db = 20
g_arr = prototype_gs(power_G_db, type = 'chebyshev', n = 5, r_db = 2)
z_arr = np.array([Z_squid, 15*4/np.pi, 15*4/np.pi, 15*4/np.pi, 15*4/np.pi, 50], dtype=float)
resonator_types = ['core', 'l4','l4','l4', 'l4']
coupler_types = ['cap', 'cap', 'cap', 'cap', 'cap']

inv_corr_factors = [1,1,1,1,1]

net = calculate_network(power_G_db, g_arr, z_arr, f0, dw, L_squid, printout=False)

net.gen_net_by_type(resonator_types, coupler_types, inv_corr_factors, draw=True)

fig, ax = plt.subplots()
ax.set_title('Gain by pump power and filter type')
ax.set_title('Gain by pump power and filter type')
ax.grid()

f_arr_GHz = np.linspace(f0/1e9-1, f0/1e9+1, 10001)
delta = 0e-3
f_p_GHz = 2*f0/1e9-delta

# plt.figure()
# f_test = 2*np.pi*f_arr_GHz*1e9
# s_par_test = abcd_to_s(np.moveaxis(net.ABCD_methods[1](f_test,f_test), 0,-1), 50)
# # plt.plot(f_test/1e9/2/np.pi,np.angle(s_par_test[0,0])*180/np.pi)
# plt.plot(f_test/1e9/2/np.pi,20*np.log10(np.abs(s_par_test[0,0])))
# plt.show()
# breakpoint()
d = net.draw_circuit(l = 2)
fig = net.plot_scattering(f_arr_GHz,f_p_GHz,
                        linestyle = 'solid',
                        fig = fig,
                        primary_color = 'b',
                        label_prepend='pumped mutual ');


# del_fig, del_ax = plt.subplots()
# omega_arr = net.omega_plot_arr
# del_arr = -np.gradient(np.unwrap(np.arctan2(net.Smtx_j0[0,0,:].imag, net.Smtx_j0[0,0,:].real)))/np.gradient(omega_arr)*1e9
# del_ax.plot(omega_arr/2/np.pi, del_arr)
# del_ax.grid()

plt.show()
breakpoint()