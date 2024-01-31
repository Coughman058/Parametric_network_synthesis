'''
Create infrastructure to import HFSS CSV results, and tell me what transmission line, capacitor, or in general whatever parameter I need to get what I want is.

Then create a framework that can fit integrated HFSS network results and compare them to the analytical network
'''

import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def import_HFSS_csv(filename, display = False):
    '''
    imports an HFSS csv file and returns a pandas dataframe
    '''
    file = pd.read_csv(filename)
    if display:
        print("importing", filename)
        print(file)
    return file

def interpolate_nd_hfss_mgoal_res(df,
                                  verbose = True,
                                  exclude_columns = None,
                                  dep_var_num = 1):
    '''
    Expects a pandas dataframe with columns var1name, var2name, etc and goalname, outputs a function which will interpolate
    the goalname as a function of all varnames
    '''
    varnames = np.array([df.columns[i] for i in range(len(df.columns))])
    ivarnames = varnames[0:-dep_var_num]
    ivar_num = ivarnames.size
    goalnames = varnames[-dep_var_num:]
    if verbose:
        print(f"Interpolating {goalnames} as function of {varnames[0:-dep_var_num]}")
    interpfuncs = []

    for i, goalname in enumerate(goalnames):
        ivarcpls = df.to_numpy()[:,0:ivar_num]
        if exclude_columns is not None:
            print("old cols:", ivarcpls)
            ivarcpls = np.delete(ivarcpls, exclude_columns, axis = 1)
            print("new cols:", ivarcpls)
        dvarlist = df.to_numpy()[:,-dep_var_num+i]
        try:
            interpfunc = LinearNDInterpolator(ivarcpls, dvarlist)
        except ValueError: #this happens if it is a 1d sweep, then we need interp1d
            interpfunc = interp1d(ivarcpls[:,0], dvarlist)
        interpfuncs.append(interpfunc)
    return interpfuncs



def display_interpolation_result(interpfuncs, df, exclude_column = None, optimization = [], primary_cols = [0,1], plot_constrained = False):
    '''
    Plots the interpolation function result and the points that it was swept over
    '''

    dep_var_num = len(interpfuncs)
    if dep_var_num <=2:
        fig, axs = plt.subplots(nrows=1, ncols=len(interpfuncs))
        if dep_var_num == 1:
            axs = [axs]
    # print(f"dep_var=_num: {dep_var_num}")
    '''
    always ignore a column first, then talk about the constraints on the rest
    !TODO: find a way to unify these two cases
    '''
    varnames = np.array([df.columns[i] for i in range(len(df.columns))])
    if exclude_column is not None:
        varnames = np.delete(varnames, exclude_column)

    ivarnames = varnames[0:-dep_var_num]
    goalnames = varnames[-dep_var_num:]
    for j in range(dep_var_num):
        interpfunc = interpfuncs[j]
        print(j)
        ax = axs[j]
        goalname = goalnames[j]
        opt_res = optimization[j]
        # print("Displaying interpolation result for", goalname, "as function of", ivarnames)
        try:
            nd = interpfunc.points[0].size
        except AttributeError: #interp1d vs linearNdinterpolator
            nd = 1
        if nd == 1: #1d sweep
            x = np.linspace(np.min(interpfunc.x), np.max(interpfunc.x), 1001)
            y = interpfunc(x)
            ax.plot(x,y)
            ax.set_xlabel(varnames[0])
            ax.set_ylabel(goalname)
            if len(optimization) > 0:
                print(opt_res)
                ax.scatter(opt_res[0], interpfunc(opt_res[0]), color='r')
                ax.set_title(goalname + f'\nopt is {interpfunc(opt_res)}\nat {opt_res}')
                ax.grid()
            else:
                ax.set_title(goalname)

        elif nd == 2:
            x = np.linspace(np.min(interpfunc.points[:,0]), np.max(interpfunc.points[:,0]), 101)
            y = np.linspace(np.min(interpfunc.points[:,1]), np.max(interpfunc.points[:,1]), 101)
            X, Y = np.meshgrid(x,y)
            Z = interpfunc(X,Y)
            im = ax.contourf(X,Y,Z)
            ax.scatter(interpfunc.points[:,0], interpfunc.points[:,1], color = 'k')
            ax.set_xlabel(varnames[0])
            ax.set_ylabel(varnames[1])
            plt.colorbar(im)
            if len(optimization) > 0:
                ax.scatter(opt_res[0], opt_res[1], color='r')
                ax.set_title(goalname + f'\nopt is {interpfunc(opt_res)}\nwith {opt_res}')
            else:
                ax.set_title(goalname)
        else:
            print ("input more than three dimensional")
            print(f"Optimization results for {goalname} = {interpfunc(opt_res)}: ")
            print(f"{ivarnames} = {opt_res}")
            if plot_constrained:
                print(f"Plotting constrained sweep for simultaneous optimization")
                constrained_vars = [varnames[i] for i in range(len(varnames)) if i not in primary_cols and i < len(opt_res-dep_var_num)]
                constrained_vals = [opt_res[i] for i in range(len(opt_res)) if i not in primary_cols and i < len(opt_res-dep_var_num)]
                print(f"Constrained variables: {constrained_vars} = {constrained_vals}")
                def constrained_interpfunc(x1, x2, primary_cols = primary_cols):
                    input_arr = []
                    for input_col in range(np.shape(opt_res)[-1]):
                        if input_col == primary_cols[0]:
                            input_arr.append(x1)
                        elif input_col == primary_cols[1]:
                            input_arr.append(x2)
                        else:
                            input_arr.append(np.ones(x1.shape)*opt_res[input_col])
                    input_arr = np.array(input_arr)
                    print(np.shape(input_arr))
                    return interpfunc(*input_arr)

                x = np.linspace(np.min(interpfunc.points[:, primary_cols[0]]), np.max(interpfunc.points[:, primary_cols[0]]), 101)
                y = np.linspace(np.min(interpfunc.points[:, primary_cols[1]]), np.max(interpfunc.points[:, primary_cols[1]]), 101)
                X, Y = np.meshgrid(x, y)
                Z = constrained_interpfunc(X, Y)
                im = ax.contourf(X, Y, Z)
                print("\n\nplotted image\n\n")
                print(Z)
                ax.scatter(interpfunc.points[:, primary_cols[0]], interpfunc.points[:, primary_cols[1]], color='k')
                ax.set_xlabel(varnames[primary_cols[0]])
                ax.set_ylabel(varnames[primary_cols[1]])
                plt.colorbar(im)
                if len(optimization) > 0:
                    ax.scatter(opt_res[primary_cols[0]], opt_res[primary_cols[1]], color='r')
                    ax.set_title(goalname + f'\nopt is {interpfunc(opt_res)}\nwith {opt_res}\nconstrained by {constrained_vars} =\n{constrained_vals}')
                else:
                    ax.set_title(goalname)

    return fig, axs

def optimize_for_goal(interpfuncs, goal_vals, p0 = None, all_weights = None):
    '''
    Finds the optimal value of the independent variables to get the goal_val
    '''
    res_arr = []
    if all_weights is None:
        all_weights = np.ones(len(interpfuncs))
    def goal_diff(x):
        to_sum = []
        for i, interpfunc in enumerate(interpfuncs):
            goal = goal_vals[i]
            cost = np.sum(np.abs(interpfunc(*x) - goal)*all_weights[i]/goal)
            to_sum.append(cost)
        return np.sum(to_sum)

    if p0 is None:
        try: #nd guesser
            p0 = interpfuncs[0].points[len(interpfuncs[0].points) // 2]
        except AttributeError: #1d guesser
            p0 = interpfuncs[0].x[len(interpfuncs[0].x) // 2]
    res = minimize(goal_diff, p0, method='nelder-mead', options = {'tol': 1e-6})
    res_arr.append(res.x)
    return res_arr
