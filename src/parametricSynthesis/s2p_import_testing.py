from parametricSynthesis import network_tools, drawing_tools, simulation_tools
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import schemdraw.elements as elm
import schemdraw
import plotly.express as px
import pandas as pd
from parametricSynthesis.simulation_tools.s2p_tuning import sweep_core_inductance_and_inversion_rate_from_filelist as sweep_from_filelist

filename_baseline = r'C:\Users\Ryan\Documents\GitHub\Parametric_network_synthesis\testing\s2p_files\CC_425.s2p'

# L_vals = np.arange(0.6, 1.5, 0.05)*1e-9
L_vals = np.array([1.2])*1e-9
J_vals = np.array([0.001, 0.025, 0.035, 0.045])
# J_vals = np.arange(0.01, 0.06, 0.005)

omega_arr = np.concatenate([np.linspace(2e9, 4e9, 1001)*2*np.pi, np.linspace(6e9, 8e9, 1001)*2*np.pi])

baseline_dict = dict(zip([filename_baseline], [0]))

total_data_baseline, HFSS_sweep_sims = sweep_from_filelist(baseline_dict, 'baseline', L_vals, J_vals, 2*np.pi*7*1e9, 0.35, omega_arr = omega_arr)

HFSS_sweep_to_examine = HFSS_sweep_sims[0]
HFSS = HFSS_sweep_to_examine

mode_find_omega_arr = np.linspace(1e9, 14e9, 1001)*2*np.pi
# res = HFSS.find_modes_from_input_impedance(1.2e-9, mode_find_omega_arr)

L_test_arr = np.array([1.2e-9, 1.2e-9])
J_test_arr = np.array([0.0175, 0.0175])
omega_test_arr = np.array([7e9, 7e9])*2*np.pi
test_ABCD_active = HFSS.evaluate_ABCD_mtx(L_test_arr, J_test_arr, omega_test_arr, active = True)
test_ABCD_passive = HFSS.evaluate_ABCD_mtx(L_test_arr, J_test_arr, omega_test_arr, active = False)

Z = HFSS.filterZmtxs
print(Z.shape)
Z0 = 50
p2_input_impedance = Z[1,1, :]-Z[1,0, :]*Z[0,1, :]/(Z[0,0, :]+Z0*np.ones_like(Z[0,0, :]).astype(complex))
p2_input_admittance = 1/p2_input_impedance