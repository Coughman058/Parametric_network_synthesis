'''
Test importing an HFSS sweep and deciphering parameters
'''
# from parametricSynthesis.testing.parameter_interp_files.NIST_2024
from parametricSynthesis.simulation_tools import parameter_interpolation as pi
import matplotlib.pyplot as plt
import numpy as np
plt.ioff()

# amp_cap_cpld_core_NIST = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\20240207_Giga_core_sweep.csv")
resonators_3d_NIST = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\NIST_2024\L_shaped_crossover_updatedconvergence_tlineZo_freq_width_gap_resheight.csv")
core_NIST = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\NIST_2024\Core_LCsweep_freq_Zres.csv")

dep_var_num = 2
SMAA_cap_cpld_core = False
amp_tline_res = True
amp_core_res = False
#amp core:
# constrained optimization, Nd
if SMAA_cap_cpld_core:
    cols_to_exclude = []
    primary_cols = [0,1]
    goals = [7e9, 20]
    import_file = amp_cap_cpld_core_NIST
    #find a pair here, then take the length and hold it constant
    interpfuncs = pi.interpolate_nd_hfss_mgoal_res(import_file, exclude_columns = cols_to_exclude, dep_var_num=dep_var_num)
    print("Interpolation successful")
    opt_res_vals = pi.optimize_for_goal(interpfuncs, goals)[0]
    display_funcs = interpfuncs
    print("Optimization successful")
    print("opt_res_arr",opt_res_vals)
    fig, axs = pi.display_interpolation_result(display_funcs, import_file,
                                               optimization=[opt_res_vals, opt_res_vals],
                                               exclude_column = cols_to_exclude,
                                               primary_cols = primary_cols,
                                               plot_constrained=True)
if amp_tline_res:
    cols_to_exclude = []
    primary_cols = [0, 1]
    goals = [8.9204e9, 20]  # res1
    # goals = []
    import_file = resonators_3d_NIST
    # find a pair here, then take the length and hold it constant
    interpfuncs = pi.interpolate_nd_hfss_mgoal_res(import_file, exclude_columns=cols_to_exclude,
                                                   dep_var_num=dep_var_num)
    print("Interpolation successful")
    opt_res_vals = pi.optimize_for_goal(interpfuncs, goals)[0]
    display_funcs = interpfuncs
    print("Optimization successful")
    print("opt_res_arr", opt_res_vals)
    fig, axs = pi.display_interpolation_result(display_funcs, import_file,
                                               optimization=[opt_res_vals, opt_res_vals],
                                               exclude_column=cols_to_exclude,
                                               primary_cols=primary_cols,
                                               plot_constrained=True)
if amp_core_res:
    cols_to_exclude = []
    primary_cols = [0, 1]
    #3-pole, dw = 0.1:
    L, C = 0.4e-9, 1.346e-12
    #dw = 0.15:
    L, C = 0.4e-9, 1.265e-12
    #dw = 0.2:
    L, C = 0.4e-9, 1.175e-12
    goals = [1/np.sqrt(L*C)/2/np.pi, np.sqrt(L/C)]  # res1
    print("goals", goals)
    # goals = []
    import_file = core_NIST
    # find a pair here, then take the length and hold it constant
    interpfuncs = pi.interpolate_nd_hfss_mgoal_res(import_file, exclude_columns=cols_to_exclude,
                                                   dep_var_num=dep_var_num, verbose = False)
    print("Interpolation successful")
    opt_res_vals = pi.optimize_for_goal(interpfuncs, goals)[0]
    display_funcs = interpfuncs
    print("Optimization successful")
    print("opt_res_arr", opt_res_vals)
    fig, axs = pi.display_interpolation_result(display_funcs, import_file,
                                               optimization=[opt_res_vals, opt_res_vals],
                                               exclude_column=cols_to_exclude,
                                               primary_cols=primary_cols,
                                               plot_constrained=True)

plt.show()
#
breakpoint()