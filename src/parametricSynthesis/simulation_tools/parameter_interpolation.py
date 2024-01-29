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
    if display:
        print("importing", filename)
        file = pd.read_csv(filename)
        print(file)

    return file

def interpolate_nd_hfss_mgoal_res(df, verbose = True, dep_var_num = 1):
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
        dvarlist = df.to_numpy()[:,ivar_num+i]
        interpfunc = LinearNDInterpolator(ivarcpls, dvarlist)
        interpfuncs.append(interpfunc)
    return interpfuncs



def display_interpolation_result(interpfuncs, df, optimization = []):
    '''
    Plots the interpolation function result and the points that it was swept over
    '''

    fig, axs = plt.subplots(nrows = 1, ncols = len(interpfuncs))
    dep_var_num = len(interpfuncs)
    if dep_var_num == 1:
        axs = [axs]
    # print(f"dep_var=_num: {dep_var_num}")
    varnames = np.array([df.columns[i] for i in range(len(df.columns))])
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
        else:
            raise Exception("Can't display interpolation result for more than 2 dimensions")

        if len(optimization) > 0:
            ax.scatter(opt_res[0], opt_res[1], color='r')
            ax.set_title(goalname+f'\nopt for {interpfunc(opt_res)}\nat {opt_res}')
        else:
            ax.set_title(goalname)

    return fig, axs

def optimize_for_goal(interpfunc, goal_val, p0 = None):
    '''
    Finds the optimal value of the independent variables to get the goal_val
    '''
    def goal_diff(x):
        return np.abs(interpfunc(*x) - goal_val)

    #most of these functions are monotonic, so we can just use Nelder-Mead from the middle of the interpolation function
    if p0 is None:
        p0 = interpfunc.points[np.size(interpfunc.points,0)//2]
    res = minimize(goal_diff, p0, method = 'Nelder-Mead')
    return res.x
