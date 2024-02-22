from parametricSynthesis.network_tools.network_synthesis import (get_active_network_prototypes,
                                                                 get_passive_network_prototypes,
                                                                 calculate_network)
from parametricSynthesis.network_tools.prototype_calc import prototype_gs
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import cm
import proplot as pplt
viridis = cm.get_cmap('viridis', 12)
rainbow = cm.get_cmap('rainbow', 12)
winter = cm.get_cmap('winter', 12)
magma = cm.get_cmap('magma', 12)
hot = cm.get_cmap('hot', 12)
brg = cm.get_cmap('brg', 12)
n_bnds = [2,8]
def var_to_color(var, var_bnds = [0,1], cmap = magma):
    var_range = var_bnds[1] - var_bnds[0]
    return cmap(0.8*((var-var_bnds[0])/(var_range))+0.1)

filt_types = ['chebyshev','legendre']

figwidth = '1in'
dw = 0.1
ymin, ymax = 0, 6*0.2/dw
path_to_folder = r'C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Figures'
n_arr = [2,3,4,5,6,7,8]
r_db_arr = np.linspace(0.1, 3, 101)
g_swp = []
Q_arr = []
fig, axs = pplt.subplots(ncols = len(filt_types), refwidth=figwidth, refaspect=0.5)
for i, filt_type in enumerate(filt_types):
    ax = axs[i]
    for n in n_arr:
        g_swp_append = []
        Q_arr_append = []
        for r_dB in r_db_arr:
            g_arr = prototype_gs(20, type = filt_type, n = n, r_db = r_dB, debug = False)
            g_swp_append.append(g_arr)
            Q_arr_append.append(g_arr[1]/dw)
        g_swp.append(g_swp_append)
        Q_arr.append(Q_arr_append)
        if i == 0:
            ax.plot(r_db_arr, Q_arr_append, label = f'n = {n}', color = var_to_color(n, n_bnds))
        else:
            ax.plot(r_db_arr, Q_arr_append, color = var_to_color(n, n_bnds))

    ax.set_title(f'{filt_type.capitalize()} Filter\n$w={dw}$')
    ax.set_xlabel('Ripple (dB)')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Q')
    ax.hlines([3.5], np.min(r_db_arr), np.max(r_db_arr), linestyle = 'dashed', color = 'red')



fig.legend(ncols = 1, loc = 'right', title = 'Filter order')
fig.savefig(path_to_folder+'\\'+rf'{filt_types}_w{int(dw*100)}pct_Q_vs_ripple.png')

#sketch an inverter net
f0 = 7e9
w0 = 2 * np.pi * f0
dw = 0.1
L_squid = 0.5e-9
# g_arr = np.array([1, 1.5963, 1.0967, 1.5963, 1])
#^^^pozar example
g_arr = prototype_gs(20, type = 'chebyshev', n = 4, r_db = 1)
z_arr = np.array([0, 35*4/np.pi, 35*4/np.pi, 35*4/np.pi, 50], dtype=float)
tline_corr_factor = 0.982

f_arr_GHz = np.linspace(f0 / 1e9 - 0.5, f0 / 1e9 + 0.5, 401)
net = calculate_network(g_arr, z_arr, f0, dw, L_squid, printout=False, inv_corr_factor=tline_corr_factor)
from parametricSynthesis.drawing_tools.sketching_functions import sketch_ideal_inverter_net
d = sketch_ideal_inverter_net(net)
breakpoint()
plt.show()




