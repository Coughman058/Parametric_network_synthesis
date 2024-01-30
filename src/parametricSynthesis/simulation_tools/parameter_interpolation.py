'''
Create infrastructure to import HFSS CSV results, and tell me what transmission line, capacitor, or in general whatever parameter I need to get what I want is.

Then create a framework that can fit integrated HFSS network results and compare them to the analytical network
'''

import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
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
        interpfunc = LinearNDInterpolator(ivarcpls, dvarlist)
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
        if interpfunc.points[0].size == 1: #1d sweep
            x = np.linspace(np.min(interpfunc.points), np.max(interpfunc.points), 1001)
            y = interpfunc(x)
            ax.plot(x,y)
            ax.set_xlabel(varnames[0])
            ax.set_ylabel(goalname)
            if len(optimization) > 0:
                ax.scatter(opt_res[0], opt_res[1], color='r')
                ax.set_title(goalname + f'\nopt is {interpfunc(opt_res)}\nat {opt_res}')
            else:
                ax.set_title(goalname)

        elif interpfunc.points[0].size == 2:
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
                constrained_vars = [varnames[i] for i in range(len(varnames)) if i not in primary_cols]
                constrained_vals = [opt_res[i] for i in range(len(opt_res)) if i not in primary_cols]
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

def optimize_for_goal(interpfuncs, goal_vals, p0 = None, optimize_all = False, all_weights = None, primary_cols = [0,1]):
    '''
    Finds the optimal value of the independent variables to get the goal_val
    '''
    res_arr = []
    if optimize_all == False:
        for interpfunc, goal_val in zip(interpfuncs, goal_vals):
            def goal_diff(x):
                return np.abs(interpfunc(*x) - goal_val)
            if p0 is None:
                p0 = interpfunc.points[interpfunc.points.size//2]
            res = minimize(goal_diff, p0, method='nelder-mead')
            res_arr.append(res.x)
    else:
        if all_weights is None:
            all_weights = np.ones(len(interpfuncs))
        def goal_diff(x):
            # print("all_weights[0]: ", all_weights[0])
            to_sum = []
            for i, interpfunc in enumerate(interpfuncs):
                cost = np.sum(np.abs(interpfunc(*x) - goal_vals[i])/goal_vals[i])
                to_sum.append(cost)
            return np.sum(to_sum)

        if p0 is None:
            p0 = interpfuncs[0].points[len(interpfuncs[0].points) // 2]
        res = minimize(goal_diff, p0, method='nelder-mead', options = {'tol': 1e-6})
        # print(res
        res_arr.append(res.x)
    return res_arr
