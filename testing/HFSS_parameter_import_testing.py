'''
Test importing an HFSS sweep and deciphering parameters
'''

from parametricSynthesis.simulation_tools import parameter_interpolation as pi
import matplotlib.pyplot as plt
plt.ioff()

testfile_3d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\20240129_tlin_sweep_sa_no_wb_oneloop.csv")
testfile_3d_with_fixed_gap = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\20240129_tlin_sweep_sa_oneloop_fixed_gap.csv")
testfile_2d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\tlineZo_cpw_width_gap_freq_Sa_NoWirebond.csv")
testfile_1d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\tlineZo_Sa_NoWirebond.csv")
dep_var_num = 2



#independent optimization
# cols_to_exclude = [1]
# interpfuncs = pi.interpolate_nd_hfss_mgoal_res(testfile_1d, exclude_columns = col_to_exclude, dep_var_num=dep_var_num)
# opt_res_arr = []
# goals = [60, 7e9]
# for i, interpfunc in enumerate(interpfuncs):
#     opt_res_x = pi.optimize_for_goal([interpfunc], goals[i], optimize_all = False)[0]
#     opt_res_arr.append(opt_res_x)
# fig, axs = pi.display_interpolation_result(interpfuncs, testfile_1d, optimization=opt_res_arr, exclude_column = cols_to_exclude)
# plt.show()

#simultaneous optimization, 2d
# goals = [80, 7e9]
# cols_to_exclude = [0,1]
# primary_cols = [0,2]
# #find a pair here, then take the length and hold it constant
# interpfuncs = pi.interpolate_nd_hfss_mgoal_res(testfile_3D_with_fixed_gap, exclude_columns = cols_to_exclude, dep_var_num=dep_var_num)
# print("Interpolation successful")
# opt_res_vals = pi.optimize_for_goal(interpfuncs, goals, optimize_all=True)[0]
# # opt_res_vals = [res.x for res in opt_res]
# display_funcs = interpfuncs
# print("Optimization successful")
# print("opt_res_arr",opt_res_vals)
# fig, axs = pi.display_interpolation_result(display_funcs, testfile_3D_with_fixed_gap,
#                                            optimization=[opt_res_vals, opt_res_vals],
#                                            exclude_column = cols_to_exclude)
# plt.show()

#constrained optimization, 3d
cols_to_exclude = [1]
primary_cols = [0,2]
goals = [40, 7e9]
#find a pair here, then take the length and hold it constant
interpfuncs = pi.interpolate_nd_hfss_mgoal_res(testfile_3d, exclude_columns = cols_to_exclude, dep_var_num=dep_var_num)
print("Interpolation successful")
opt_res_vals = pi.optimize_for_goal(interpfuncs, goals, optimize_all=True)[0]
display_funcs = interpfuncs
print("Optimization successful")
print("opt_res_arr",opt_res_vals)
fig, axs = pi.display_interpolation_result(display_funcs, testfile_3d,
                                           optimization=[opt_res_vals, opt_res_vals],
                                           exclude_column = cols_to_exclude,
                                           primary_cols = primary_cols,
                                           plot_constrained=True)
plt.show()

breakpoint()