'''
Test importing an HFSS sweep and deciphering parameters
'''

from parametricSynthesis.simulation_tools import parameter_interpolation as pi
import matplotlib.pyplot as plt
plt.ioff()

# testfile_3d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\20240129_tlin_sweep_sa_no_wb_oneloop.csv")
# testfile_3d_with_fixed_gap = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\20240129_tlin_sweep_sa_oneloop_fixed_gap.csv")
# testfile_2d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\tlineZo_cpw_width_gap_freq_Sa_NoWirebond.csv")
# testfile_1d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\tlineZo_Sa_NoWirebond.csv")
resonators_3d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\best\res_sweeps.csv")
inverters_3d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\best\inv_sweeps.csv")
resonator_core = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\best\core_res_cap_sweep.csv")
resonators_3d_hf = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\best\res_sweeps_higher_freq.csv")
amp_cap_cpld_core = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\20240207_Giga_core_sweep.csv")
resonators_3d_NIST = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\best\res_sweeps_NIST.csv")

dep_var_num = 2
SMAA_cap_cpld_core = False
amp_tline_res = False
amp_core_res = False
#amp core:
# constrained optimization, Nd
if SMAA_cap_cpld_core:
    cols_to_exclude = []
    primary_cols = [0,1]
    goals = [7e9, 20]
    import_file = amp_cap_cpld_core
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
    # goals = [35, 7.6364e9]#res1
    goals = [35, 8.5586e9]  # res2
    import_file = resonators_3d_hf
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
    # independent optimization
    dep_var_num = 1
    cols_to_exclude = []
    primary_cols = [0]
    interpfuncs = pi.interpolate_nd_hfss_mgoal_res(resonator_core, exclude_columns = cols_to_exclude, dep_var_num=dep_var_num)
    opt_res_arr = []
    goals = [7.1647e9]
    opt_res_x = pi.optimize_for_goal(interpfuncs, goals)[0]
    opt_res_arr.append(opt_res_x)
    fig, axs = pi.display_interpolation_result(interpfuncs, resonator_core,
                                               optimization=opt_res_arr,
                                               exclude_column = cols_to_exclude,
                                               primary_cols = primary_cols)

plt.show()
#
breakpoint()