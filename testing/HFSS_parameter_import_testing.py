'''
Test importing an HFSS sweep and deciphering parameters
'''

from parametricSynthesis.simulation_tools import parameter_interpolation as pi
import matplotlib.pyplot as plt
plt.ioff()

testfile_2d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\tlineZo_cpw_width_gap_freq_Sa_NoWirebond.csv", display = True)
# testfile_1d = pi.import_HFSS_csv(r"C:\Users\Hatlab-RRK\Documents\GitHub\Parametric_network_synthesis\testing\Parameter_interp_files\tlineZo_Sa_NoWirebond.csv", display = True)
testfile = testfile_2d
dep_var_num = 2
interpfuncs = pi.interpolate_nd_hfss_mgoal_res(testfile, dep_var_num=2)
opt_res_arr = []
goals = [60, 7e9]
for i, interpfunc in enumerate(interpfuncs):
    opt_res_x = pi.optimize_for_goal(interpfunc, goals[i])
    opt_res_arr.append(opt_res_x)
fig, axs = pi.display_interpolation_result(interpfuncs, testfile, optimization=opt_res_arr)
plt.show()
breakpoint()