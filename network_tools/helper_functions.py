import numpy as np

calculate_middle_inverter_constant = lambda dw, gi, gj, zi, zj: np.sqrt(dw**2/(gi*gj*zi*zj))
calculate_outer_inverter_constant = lambda dw, gi, gj, zi, zj: np.sqrt(dw/(gi*gj*zi*zj))
calculate_last_resonator_impedance = lambda dw, z0, g_last, g_next_to_last: dw*z0/(g_last*g_next_to_last)
calculate_PA_impedance = lambda ZPA_res, g0, g1, dw: g0*g1*ZPA_res/dw
calculate_middle_beta = lambda dw, gi,gj, gamma_0: dw/(2*gamma_0*np.sqrt(gi*gj))
calculate_PA_beta = lambda g_arr: g_arr[-1]*g_arr[-2]/(2*g_arr[0]*g_arr[1])
