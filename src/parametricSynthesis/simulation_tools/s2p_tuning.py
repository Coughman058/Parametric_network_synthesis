'''
The goal of this module is to implement live plotly analysis of HFSS-exported s2p files in a google colaboratory notebook.
'''

import pandas as pd
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from ipywidgets import widgets
from ..simulation_tools.HFSS_analysis import interpolated_network_with_inverter_from_filename as net_with_inverter
# L_vals = np.arange(0.5, 1.1, 0.05)*1e-9
# J_vals = np.arange(0.01, 0.03, 0.001)

def sweep_core_inductance_and_inversion_rate_from_filelist(filenames_dict_with_vals:dict,
                                                           sweep_parameter_name:str,
                                                           L_vals:np.ndarray,
                                                           J_vals:np.ndarray,
                                                           omega0_val:float,
                                                           dw_val:float,
                                                           omega_arr = np.linspace(5e9,9e9,1001)*2*np.pi):
    '''
    takes in a dictionary of filenames and sweep values for a 1D sweep of s2p files,
    and returns a list of functions that return the S matrix of the network
    as a function of the core inductance and inversion rate of the inverter
    :param filenames_CC:
    :return:
    '''
    filenames_sweep = filenames_dict_with_vals.keys()
    sweep_vals = filenames_dict_with_vals.values()
    L_array_sym, J_array_sim = sp.symbols("L_pa, J_pa")
    HFSS_sweep_sims = [net_with_inverter(filename=filename,
                                       L_sym=L_array_sym,
                                       inv_J_sym=J_array_sim,
                                       dw=dw_val,
                                       omega0_val=omega0_val) for filename in filenames_sweep]
    #bundle the data into a pandas dataframe for plotly
    # print("DEBUG: L_vals in plotly_1D_sweep is: ", L_vals)
    data_list = []
    for L in L_vals:
      for J in J_vals:
        for sweep_val, HFSS in zip(sweep_vals, HFSS_sweep_sims):
          L_arr = L*np.ones(omega_arr.size)
          Jpa_arr = J*np.ones(omega_arr.size)
          Smtx_res = HFSS.evaluate_Smtx(L_arr, Jpa_arr, omega_arr)
          data = pd.DataFrame({'Signal Frequency (GHz)': omega_arr/2/np.pi/1e9,
                                'Array Inductance (nH)': L_arr*1e9,
                                'Inversion Rate': Jpa_arr,
                                sweep_parameter_name: sweep_val*np.ones(omega_arr.size),
                                'S11magDB': 20*np.log10(np.abs(Smtx_res[0,0])),
                               'S21magDB': 20*np.log10(np.abs(Smtx_res[1,0]))})
          data_list.append(data)

    total_data = pd.concat(data_list, ignore_index = True, axis = 0)

    return total_data


def plotly_1D_sweep(total_data, sweep_val_name = 'HFSS sweep parameter', x_axis_name = 'Signal Frequency (GHz)', y_axis_names = ['S11magDB', 'S21magDB'], yscale = [-5, 35]):
    L_vals = np.unique(total_data['Array Inductance (nH)'])
    J_vals = np.unique(total_data['Inversion Rate'])
    sweep_vals = np.unique(total_data[sweep_val_name])
    Inductance = widgets.RadioButtons(
        options=L_vals,
        value=L_vals[0],
        description='Array Inductance (nH)',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True
    )

    Inversion = widgets.RadioButtons(
        options=J_vals,
        value=J_vals[0],
        description='Inversion Rate',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True
    )

    Sweep = widgets.RadioButtons(
        options=sweep_vals,
        value=sweep_vals[0],
        description=f'{sweep_val_name} sweep',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True
    )

    container = widgets.HBox(children=[Inductance, Inversion, Sweep])

    # Assign an empty figure widget with two traces
    trace1 = go.Line(x=total_data[x_axis_name], opacity=1, name='SignalGain')
    trace2 = go.Line(x=total_data[x_axis_name], opacity=1, name='IdlerGain')
    g = go.FigureWidget(data=[trace1, trace2],
                        layout=go.Layout(
                            title=dict(
                                text='Power Gain in dB'
                            ),
                        ))

    def response(change):
        temp_df = total_data.loc[
            (total_data['Array Inductance (nH)'] == Inductance.value) & (total_data['Inversion Rate'] == Inversion.value) & (
                        total_data[sweep_val_name] == Sweep.value)]
        x1 = temp_df[x_axis_name]
        y1 = temp_df[y_axis_names[0]]
        x2 = temp_df[x_axis_name]
        y2 = temp_df[y_axis_names[1]]
        with g.batch_update():
            g.data[0].x = x1
            g.data[0].y = y1
            g.data[1].x = x2
            g.data[1].y = y2
            g.layout.xaxis.title = x_axis_name
            # g.layout.yaxis.title = y_axis_names
            g.layout.yaxis.range = yscale

    # Observe change in slider and update
    slider_list = [Inductance, Inversion, Sweep]

    return slider_list, response, widgets.VBox([container,g])
    #what you need to do to make this work in google colab:
    # from google.colab import output
    # output.enable_custom_widget_manager()
    #
    # widgets.VBox([container,
    #               g])