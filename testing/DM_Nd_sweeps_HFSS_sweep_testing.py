import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from scipy.interpolate import LinearNDInterpolator, interp1d
import sympy as sp
import parametricSynthesis
from parametricSynthesis.simulation_tools.HFSS_analysis import NdHFSSSweepOptimizer
from parametricSynthesis.network_tools.network_synthesis import Network

active_network_prototypes = parametricSynthesis.network_tools.network_synthesis.get_active_network_prototypes()
active_network_prototypes
f0 = 7e9
w0 = 2*np.pi*f0
dw = 0.075
L_squid = 0.5e-9
Z_squid = w0*L_squid
g_arr = active_network_prototypes['N2_Cheby_20dB_R05']
z_arr = np.array([0,60,50], dtype = float)
tline_corr_factor = 0.85
f_arr_GHz = np.linspace(f0/1e9-0.7,f0/1e9+0.7,201)
ideal_net = parametricSynthesis.network_tools.network_synthesis.calculate_network(g_arr, z_arr, f0, dw, L_squid, printout = False)
fig, ax = plt.subplots()
ax.set_title('Gain by pump power and filter type')
ax.grid()
#pumped mutual
ideal_net.gen_net_by_type('tline_cpld_l4', active = True, core_inductor = False, method = 'pumped_mutual',
                  tline_inv_Z_corr_factor = tline_corr_factor, use_approx = False) #0.945
fig = ideal_net.plot_scattering(f_arr_GHz,
                            linestyle = 'solid',
                            fig = fig,
                            vary_pump = False,
                            method = 'pumped_mutual',
                            focus = True,
                            primary_color = 'b',
                            label_prepend='pumped mutual ',
                            debug = True);


filename_2D = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\integrated_modelling\01_DM_cap_finger_length_and_inv1_cpw_width.csv"
# filename_3D = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\integrated_modelling\02_DM_cap_finger_length_and_inv1_cpw_width_and_inv2_cpw_width.csv"
analyzer = NdHFSSSweepOptimizer(filename_2D)
analyzer.optimize_params(ideal_net)
