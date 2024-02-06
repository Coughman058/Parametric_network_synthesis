import os

from parametricSynthesis import network_tools, drawing_tools, simulation_tools
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from parametricSynthesis.simulation_tools.s2p_tuning import sweep_core_inductance_and_inversion_rate_from_filelist as sweep_from_filelist
'''
objective of this model is to take in some result from HFSS (which is not what I want) and fit it to the ideal model by 
changing all of the circuit parameters. In other words: figure out what's wrong. Then hopefully I can fix it
'''

#first generate the ideal model that we will be comparing to.
active_network_prototypes = network_tools.network_synthesis.get_active_network_prototypes()
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
net = network_tools.network_synthesis.calculate_network(g_arr, z_arr, f0, dw, L_squid, printout = False)
# net_types = ['tline_cpld_lumped', 'tline_cpld_l2', 'tline_cpld_l4']
# linestyles = ['dotted', 'dashed', 'solid']
net_types = ['tline_cpld_l4']
linestyles = ['solid']
fig, ax = plt.subplots()
ax.set_title('Gain by pump power and filter type')
ax.grid()

#pumped mutual
for net_type, linestyle in zip(net_types, linestyles):
  net.gen_net_by_type(net_type, active = True, core_inductor = False, method = 'pumped_mutual',
                      tline_inv_Z_corr_factor = tline_corr_factor, use_approx = False) #0.945
  fig = net.plot_scattering(f_arr_GHz,
                            linestyle = linestyle,
                            fig = fig,
                            vary_pump = False,
                            method = 'pumped_mutual',
                            focus = True,
                            primary_color = 'b',
                            label_prepend='pumped mutual ',
                            debug = True);
ax.legend(bbox_to_anchor = (1,1), ncol = 3)
fig.suptitle("Gain: Ideal")

#let's grab that impedance function looking from the inverter, we will use this to verify the hfss simulation
passive_Z_func_from_inv = sp.lambdify([net.omega_from_inverter],
                                      net.passive_impedance_seen_from_inverter(add_index=0).subs(net.net_subs))

'''
we want to lambdify all except the last four things on the net_subs list, 
which are the characteristic impedance, frequency substitutions and the inductor substitution
'''
initial_device_fit_guess = []
lambdify_list = []
for net_sub in net.net_subs_before_idler[:-4]:
    var, guess = net_sub
    if var not in lambdify_list:
        guess = float(guess)
        lambdify_list.append(var)
        initial_device_fit_guess.append(guess)
    #we need to turn these into lambdification variables and an array that corresponds to the initial design fit guess.
lambdify_list.append(net.net_subs_before_idler[-1][0])
theta_comp = sp.symbols("theta_comp")
lambdify_list.append(theta_comp)
initial_device_fit_guess.append(net.net_subs_before_idler[-1][1])
initial_device_fit_guess.append(np.pi)

impedance_func_to_lambdify = sp.exp(sp.I*theta_comp*net.omega_from_inverter/w0)*net.passive_impedance_seen_from_inverter(add_index=0).subs(net.net_subs_before_idler[-4:-1])

omega_arr = 2*np.pi*np.linspace(5e9, 9e9, 10001)

lambdified_z_func = sp.lambdify(lambdify_list+[net.omega_from_inverter], impedance_func_to_lambdify)
def impedance_func_to_fit(args):
    '''
    this function takes in the arguments, which are fractions of the initial guesses, and outputs the impedance function
    evaluated at the omega_arr
    '''
    input_list = []
    for i, val in enumerate(args):
        # print(val)
        input_list.append(val*np.ones_like(omega_arr))
    return lambdified_z_func(*input_list, omega_arr)

# net_sub = sp.lambdify([net.omega_from_inverter], net_sub)

filename = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\GigaV3\01_from_interp_opt_renorm.s2p"
# filename = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\CC_425.s2p"
inv2_length_dir = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\GigaV3\inv2_length_sweeps"
inv2_length_filenames = [inv2_length_dir +'\\'+ append for append in os.listdir(inv2_length_dir)]

inv2_cpw_width_dir = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\GigaV3\inv2_cpw_width_sweeps"
inv2_cpw_width_names = [append for append in os.listdir(inv2_cpw_width_dir)]
inv2_cpw_width_filenames = [inv2_cpw_width_dir +'\\'+ append for append in os.listdir(inv2_cpw_width_dir)]

inv1_cpw_width_dir = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\GigaV3\inv1_cpw_width_sweeps"
inv1_cpw_width_names = [append for append in os.listdir(inv1_cpw_width_dir)]
inv1_cpw_width_filenames = [inv1_cpw_width_dir +'\\'+ append for append in os.listdir(inv1_cpw_width_dir)]

res0_cpw_width_dir = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\GigaV3\res0_cpw_width_sweeps"
res0_cpw_width_names = [append for append in os.listdir(res0_cpw_width_dir)]
res0_cpw_width_filenames = [res0_cpw_width_dir +'\\'+ append for append in os.listdir(res0_cpw_width_dir)]

cap_finger_len_dir = r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\GigaV3\core_cap_sweeps"
cap_finger_len_names = [append for append in os.listdir(cap_finger_len_dir)]
cap_finger_len_filenames = [cap_finger_len_dir +'\\'+ append for append in os.listdir(cap_finger_len_dir)]

L_vals = np.linspace(0.3, 0.7, 9)*1e-9
J_vals = np.array([0.001, 0.025, 0.035, 0.045])
# J_vals = np.arange(0.01, 0.06, 0.005)

# omega_arr = np.concatenate([np.linspace(2e9, 4e9, 1001)*2*np.pi, np.linspace(6e9, 8e9, 1001)*2*np.pi])
# omega_arr = 2*np.pi*np.linspace(3e9, 10e9, 100001)
baseline_dict = dict(zip([filename], [0]))
inv_2_lengths = [790, 800, 810, 816, 820, 830]
inv2_length_dict = dict(zip(inv2_length_filenames, inv_2_lengths))

inv_2_cpw_widths = [7.8, 8.2, 8.6, 9.0, 9.4, 9.8]
inv2_cpw_width_dict = dict(zip(inv2_cpw_width_filenames, inv_2_cpw_widths))

inv_1_cpw_widths = [3, 4, 5, 6, 7]
inv1_cpw_width_dict = dict(zip(inv1_cpw_width_filenames, inv_1_cpw_widths))

res0_cpw_widths = [10.5, 11.5, 12.5, 13.5, 14.5]
res0_cpw_width_dict = dict(zip(res0_cpw_width_filenames, res0_cpw_widths))

cap_finger_lens = [400, 450, 500, 550, 600]
cap_finger_len_dict = dict(zip(cap_finger_len_filenames, cap_finger_lens))

total_inv2_length, HFSS_sweep_sims_inv2_length, HFSS_passive_sweep_sims_inv2_length = sweep_from_filelist(inv2_length_dict, 'inv2_length', L_vals, J_vals, 2*np.pi*7*1e9, dw, omega_arr = omega_arr)
total_inv2_cpw_width, HFSS_sweep_sims_inv2_cpw_width, HFSS_passive_sweep_sims_inv2_cpw_width = sweep_from_filelist(inv2_cpw_width_dict, 'inv2_cpw_width', L_vals, J_vals, 2*np.pi*7*1e9, dw, omega_arr = omega_arr)
total_inv1_cpw_width, HFSS_sweep_sims_inv1_cpw_width, HFSS_passive_sweep_sims_inv1_cpw_width = sweep_from_filelist(inv1_cpw_width_dict, 'inv1_cpw_width', L_vals, J_vals, 2*np.pi*7*1e9, dw, omega_arr = omega_arr)
total_res0_cpw_width, HFSS_sweep_sims_res0_cpw_width, HFSS_passive_sweep_sims_res0_cpw_width = sweep_from_filelist(res0_cpw_width_dict, 'res0_cpw_width', L_vals, J_vals, 2*np.pi*7*1e9, dw, omega_arr = omega_arr)
total_cap_finger_len, HFSS_sweep_sims_cap_finger_len, HFSS_passive_sweep_sims_cap_finger_len = sweep_from_filelist(cap_finger_len_dict, 'cap_finger_len', L_vals, J_vals, 2*np.pi*7*1e9, dw, omega_arr = omega_arr)

# HFSS_sweep_to_examine = HFSS_passive_sweep_sims[0]
# HFSS_sweep_to_examine = HFSS_sweep_sims[0]

def plot_mode_reZ_vs_inductance_vs_ideal(HFSS_sweep, L_vals, ideal_z_func, omega_arr, title = None):
    '''
    Goal is to plot the impedance seen at the inverter as a function of the inductance of the array and compare with HFSS
    '''
    Z_fig, [Z_ax_re, Z_ax_im] = plt.subplots(nrows = 1, ncols = 2, figsize = (6,4))
    Z_ax_re.set_title("Real")
    Z_ax_im.set_title("Imaginary")
    Z_fig.suptitle(title)
    Z_ax_re.plot(omega_arr / 2 / np.pi / 1e9, ideal_z_func(omega_arr).real, label='ideal',
              linestyle='dashed', color = 'k')
    Z_ax_im.plot(omega_arr / 2 / np.pi / 1e9, ideal_z_func(omega_arr).imag, label='ideal',
              linestyle='dashed', color = 'k')
    for l_squid in L_vals:
        z_from_inv = HFSS_sweep.find_p2_input_impedance(l_squid, omega_arr)
        Z_ax_re.plot(omega_arr / 2 / np.pi / 1e9, z_from_inv.real, label=f'L_squid = {np.round(l_squid*1e12)}')
        Z_ax_im.plot(omega_arr / 2 / np.pi / 1e9, z_from_inv.imag, label=f'L_squid = {np.round(l_squid*1e12)}')
    for Z_ax in [Z_ax_re, Z_ax_im]:
        Z_ax.set_xlabel('Frequency (GHz)')
        Z_ax.set_ylabel('Impedance (Ohms)')
        Z_ax.set_ylim(-200, 200)
    Z_ax_im.legend(bbox_to_anchor=(1.05, 1))
    Z_fig.tight_layout()

#look at the impedance for each parameter

#inv2_length (not much effect)
# for value, HFSS_sweep_to_examine in zip(inv_2_lengths, HFSS_sweep_sims_inv2_length):
#     plot_mode_reZ_vs_inductance_vs_ideal(HFSS_sweep_to_examine, L_vals, passive_Z_func_from_inv, omega_arr, title = value)
#inv2_cpw_width (not much difference)
# for value, HFSS_sweep_to_examine in zip(inv2_cpw_width_names, HFSS_sweep_sims_inv2_cpw_width):
#     plot_mode_reZ_vs_inductance_vs_ideal(HFSS_sweep_to_examine, L_vals, passive_Z_func_from_inv, omega_arr, title = value)
# inv1_cpw_width (notable effect, helps if cpw is narrower/higher impedance)
# for value, HFSS_sweep_to_examine in zip(inv1_cpw_width_names, HFSS_sweep_sims_inv1_cpw_width):
#     plot_mode_reZ_vs_inductance_vs_ideal(HFSS_sweep_to_examine, L_vals, passive_Z_func_from_inv, omega_arr, title = value)
# res0_cpw_width
for value, HFSS_sweep_to_examine in zip(res0_cpw_width_names, HFSS_sweep_sims_res0_cpw_width):
    plot_mode_reZ_vs_inductance_vs_ideal(HFSS_sweep_to_examine, L_vals, passive_Z_func_from_inv, omega_arr, title = value)
#cap_finger_len (significant difference but not better per se)
# for value, HFSS_sweep_to_examine in zip(cap_finger_lens, HFSS_sweep_sims_cap_finger_len):
#     plot_mode_reZ_vs_inductance_vs_ideal(HFSS_sweep_to_examine, L_vals, passive_Z_func_from_inv, omega_arr, title = value)

z_from_inv = HFSS_sweep_to_examine.find_p2_input_impedance(L_squid, omega_arr)

#do a fit to the HFSS data using the function we made earlier
def norm_impedance_goal_function(args):
    return impedance_func_to_fit([arg*initial_device_fit_guess[i] for i, arg in enumerate(args)])
def minimize_goal_function(args):
    return np.sum(np.abs(norm_impedance_goal_function(args).imag - z_from_inv.imag))
# Z_ax.plot(omega_arr/2/np.pi/1e9, impedance_func_to_fit(*np.array(initial_device_fit_guess)).real, label = 'fit, real')
# Z_ax.plot(omega_arr/2/np.pi/1e9, impedance_func_to_fit(*np.array(initial_device_fit_guess)).imag, label = 'fit, imag')
# Z_ax.plot(omega_arr/2/np.pi/1e9, norm_impedance_goal_function(*np.ones_like(initial_device_fit_guess)).real, label = 'fit, real')
# Z_ax.plot(omega_arr/2/np.pi/1e9, norm_impedance_goal_function(*np.ones_like(initial_device_fit_guess)).imag, label = 'fit, imag')
# Z_ax.legend()



# bounds = [(0.9, 1.1) for i in range(len(initial_device_fit_guess)-1)]+[(-1,1)]
# res = minimize(minimize_goal_function, np.ones_like(initial_device_fit_guess), bounds = bounds, method = 'Nelder-Mead')
# print("Success?", res.success)
# for name, val in zip(lambdify_list, res.x):
#     print("%s: %f" % (name, val))
# #plot that result as well:
# Z_ax.plot(omega_arr/2/np.pi/1e9, impedance_func_to_fit((res.x*np.array(initial_device_fit_guess))).real, label = 'fit, real')
# Z_ax.plot(omega_arr/2/np.pi/1e9, impedance_func_to_fit((res.x*np.array(initial_device_fit_guess))).imag, label = 'fit, imag')
# Z_ax.legend()
# # breakpoint()
# # mode_find_omega_arr = np.linspace(1e9, 14e9, 1001)*2*np.pi
# # roots, reY_at_roots, imYp_at_roots = HFSS.find_modes_from_input_impedance(1.2e-9, mode_find_omega_arr)
# #
# L_arr = np.linspace(0.3e-9, 0.8e-9, 11)
# modeResult = HFSS.modes_as_function_of_inductance(L_arr, omega_arr, Z0=50)
# # modeResult = HFSS.modes(omega_arr, Z0=50)
#
# fig, ax = plt.subplots()
# mode_filt = modeResult.omega_arr < 2*np.pi*8e9
# ax.scatter(modeResult.ivar_arr[mode_filt]*1e12, modeResult.omega_arr[mode_filt]/2/np.pi/1e9)
# ax.set_xlabel('L (pH)')
# ax.set_ylabel('f (GHz)')
# ax.grid()
# breakpoint()
plt.show(block = True)

