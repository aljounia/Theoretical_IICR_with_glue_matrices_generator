#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from scipy.special import comb
import argparse


def is_array_like(obj, string_is_array = False, tuple_is_array = True):
    result = hasattr(obj, "__len__") and hasattr(obj, '__getitem__') 
    if result and not string_is_array and isinstance(obj, str):
        result = False
    if result and not tuple_is_array and isinstance(obj, tuple):
        result = False
    return result

def compute_IICR_general(t,params):
    M = params["M"]
    s = params["sampling"]
    c = params["size"]
    if "tau" in params:
        tau = params["tau"]
        c = params["size"]
        return compute_non_stationary_IICR_general(t, params)
    return compute_stationary_IICR_general(M,t,s,c)

def compute_IICR_n_islands(t, params):
    n = params["n"]
    M = params["M"]
    s = params["sampling_same_island"]

    if(is_array_like(M)):
        tau = params["tau"]
        c = params["size"]

        if(not (is_array_like(tau) or is_array_like(c))):
            raise TypeError("Both 'tau' and 'size' must be array types!")
        
        if(len(M) != len(tau)):
            raise ValueError("Vectors 'M' and 'tau' must have the same length!")
        
        if(tau[0] != 0):
            raise ValueError("The time of the first event must be 0!")

        if(len(M) != len(c)):
            raise ValueError("Vectors 'M' and 'size' must have the same length!")

        return compute_piecewise_stationary_IICR_n_islands(n, M, tau, c, t, s)
    
    return compute_stationary_IICR_n_islands(n, M, t, s)

def compute_stationary_IICR_general(M,t,s,c):
    from model_ssc import SSC

    model_params = {"samplingVector" : s, "M" : M, "size" : c}
    ssc = SSC(model_params)
    return ssc.evaluateIICR(t)

def compute_stationary_IICR_n_islands(n, M, t, s=True):
    # This method computes the IICR for n-island model
    # using the exact expression of the density of T2.

    # Computing constants
    gamma = np.true_divide(M, n-1)
    delta = (1+n*gamma)**2 - 4*gamma
    alpha = 0.5*(1+n*gamma + np.sqrt(delta))
    beta =  0.5*(1+n*gamma - np.sqrt(delta))

    # Now we evaluate
    x_vector = t
    if s: # Individuals sampled in the same island
        numerator = (1-beta)*np.exp(-alpha*x_vector) + (alpha-1)*np.exp(-beta*x_vector)
        denominator = (alpha-gamma)*np.exp(-alpha*x_vector) + (gamma-beta)*np.exp(-beta*x_vector)
    else: # Individuals sampled in different islands
        numerator = beta*np.exp(-alpha*(x_vector)) - alpha*np.exp(-beta*(x_vector))
        denominator = gamma * (np.exp(-alpha*(x_vector)) - np.exp(-beta*(x_vector)))

    lambda_t = np.true_divide(numerator, denominator)

    return lambda_t

def compute_non_stationary_IICR_general(t, params):
    from model import NSSC

    M = params["M"]
    s = params["sampling"]
    c = params["size"]
    tau = params["tau"]

    scenarios = []
    for i in range(len(M)):
        thisdict = {"time" : tau[i], "M": M[i], "c": c[i]}
        scenarios.append(thisdict)

    model_params = {"samplingVector" : s, "scenario" : scenarios}
    nsnic = NSSC(model_params)
    return nsnic.evaluateIICR(t)

def compute_piecewise_stationary_IICR_n_islands(n, M, tau, c, t, s=True):
    from model import Pnisland

    sampling = []
    if(s):
        sampling = [2] + [0] * (n[0] - 1)
    else:
        sampling = [1, 1] + [0] * (n[0] - 2)

    scenarios = []
    for i in range(len(M)):
        thisdict = {"time" : tau[i], "n": n[i], "M": M[i], "c": c[i]}
        scenarios.append(thisdict)

    model_params = {"samplingVector" : sampling, "scenario" : scenarios}
    nsnic = Pnisland(model_params)
    return nsnic.evaluateIICR(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate T2 values with ms then plot the IICR')
    parser.add_argument('params_file', type=str,
                    help='the filename of the parameters')
    args = parser.parse_args()
    with open(args.params_file) as json_params:
        p = json.load(json_params)
    
    # Do the plot    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    N0 = p["scale_params"]["N0"]
    g_time = p["scale_params"]["generation_time"]
                

    # Draw the vertical lines (if specified)
    for vl in p["vertical_lines"]:
      ax.axvline(2 * N0 * g_time * vl, color='k', ls='--')
      
        
    if p["plot_params"]["plot_theor_IICR"]:
        theoretical_IICR_nisland_list = []
        theoretical_IICR_general_list = []
        T_max = np.log10(p["plot_params"]["plot_limits"][1])
        t_k = np.logspace(-1, T_max, 1000)
        t_k = np.true_divide(t_k, 2 * N0 * g_time)
        if "theoretical_IICR_nisland" in p:
            for i in range(len(p["theoretical_IICR_nisland"])):
                params = p["theoretical_IICR_nisland"][i]
                theoretical_IICR_nisland_list.append(compute_IICR_n_islands(t_k, params))
        if "theoretical_IICR_general" in p:
            for i in range(len(p["theoretical_IICR_general"])):
                params = p["theoretical_IICR_general"][i]
                theoretical_IICR_general_list.append(compute_IICR_general(t_k, params))
            
        # Plotting the theoretical IICR
        if "theoretical_IICR_nisland" in p:
            for i in range(len(p["theoretical_IICR_nisland"])):
                linecolor = p["theoretical_IICR_nisland"][i]["color"]
                line_style = p["theoretical_IICR_nisland"][i]["linestyle"]
                linewidth = p["theoretical_IICR_nisland"][i]["linewidth"]
                alpha = p["theoretical_IICR_nisland"][i]["alpha"]        
                plot_label = p["theoretical_IICR_nisland"][i]["label"]
                ax.plot(2 * N0 * g_time * t_k, N0 * theoretical_IICR_nisland_list[i],
                color=linecolor, ls=line_style, alpha=alpha, label=plot_label)
        if "theoretical_IICR_general" in p:
            for i in range(len(p["theoretical_IICR_general"])):
                linecolor = p["theoretical_IICR_general"][i]["color"]
                line_style = p["theoretical_IICR_general"][i]["linestyle"]
                linewidth = p["theoretical_IICR_general"][i]["linewidth"]
                alpha = p["theoretical_IICR_general"][i]["alpha"]        
                plot_label = p["theoretical_IICR_general"][i]["label"]
                ax.plot(2 * N0 * g_time * t_k, N0 * theoretical_IICR_general_list[i],
                color=linecolor, ls=line_style, alpha=alpha, label=plot_label)
        if "save_theor_IICR_as_file" in p:
            if p["save_theor_IICR_as_file"]:
                for i in range(len(theoretical_IICR_general_list)):
                    (time_k, theor_iicr) = (t_k,theoretical_IICR_general_list[i])
                    with open("./IICR_gen_{}_text_file.txt".format(i), "w") as f:
                        x2write = [str(2 * N0 * g_time * value) for value in t_k]
                        IICR2write = [str(N0 * value) for value in theor_iicr]
                        f.write("{}\n".format(" ".join(x2write)))
                        f.write("{}\n".format(" ".join(IICR2write)))
                for i in range(len(theoretical_IICR_nisland_list)):
                    (time_k, theor_iicr) = (t_k,theoretical_IICR_nisland_list[i])
                    with open("./IICR_nisland_{}_text_file.txt".format(i), "w") as f:
                        x2write = [str(2 * N0 * g_time * value) for value in t_k]
                        IICR2write = [str(N0 * value) for value in theor_iicr]
                        f.write("{}\n".format(" ".join(x2write)))
                        f.write("{}\n".format(" ".join(IICR2write)))


    # Plotting constant piecewise functions (if any)
    if "piecewise_constant_functions" in p:
        for f in p["piecewise_constant_functions"]:
            x = f["x"]
            y = f["y"]
            plot_label = f["label"]
            linecolor = f["color"]
            line_style = f["linestyle"]
            line_width = f["linewidth"]
            line_alpha = f["alpha"]
            ax.step(x, y, where='post', color=linecolor, ls=line_style, linewidth=line_width,
                    alpha=line_alpha, label=plot_label)

    ax.set_xlabel(p["plot_params"]["plot_xlabel"])
    ax.set_ylabel(p["plot_params"]["plot_ylabel"])
    if "y_scale" in p["plot_params"]:
        if p["plot_params"]["y_scale"] == "log":
            ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc='best')
    [x_a, x_b, y_a, y_b] = p["plot_params"]["plot_limits"]
    plt.xlim(x_a, x_b)
    plt.ylim(y_a, y_b)
    if "plot_title" in p["plot_params"]:
      ax.set_title(p["plot_params"]["plot_title"])
    if ("save_figure" in p["plot_params"]) and p["plot_params"]["save_figure"]:
        fig_name = os.path.splitext(args.params_file)[0]
        plt.savefig("{}.pdf".format(fig_name),
                        format="pdf")      
    if ("show_plot" in p["plot_params"]) and p["plot_params"]["show_plot"]:
        plt.show()

    plt.savefig('test.png')
