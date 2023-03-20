# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:36:51 2023

@author: othma
"""

import numpy as np

from src.mean_estimate.ONEOVERN.generate_mean_one_over_n import MeanGeneratorOneOverN
from src.mean_estimate.parameters import DEFAULT_PARAMS_CONSTANT_MEAN
 
def main_one_over_n(params: dict = {}) -> None:
    # Initialization parameters
    if not(params):
        params = DEFAULT_PARAMS_CONSTANT_MEAN
    
    s_val = params["s_val"]
    nbSimu = params["nbSimu"]
    nb_iter= params["nb_iter"]
    nb_episode = params["nb_episode"]
    window_size = params["window_size"]
    time_step = params["time_step"] 
    mu = params["mu"]
    alpha = params["alpha"]
    var = params["var"]
    gamma = params["gamma"]
    pctg_min = params["pctg_min"]
    print_metrics = params["print_metrics"]
    print_freq = params["print_freq"]
    h_theo = alpha * mu * time_step * np.ones(nb_iter) 
    
    
    # summary stats
    summary = {"error_window": [],
               "error_hist": [],
               "mean": [],
               "var": []}
                 
    # main routine
    for n in range(nbSimu):
        # generate forecast
        meanGen = MeanGeneratorOneOverN(s_val, nb_iter, nb_episode, window_size,
                                            time_step, mu, alpha, var, gamma, 
                                            print_metrics, pctg_min)
        meanGen.get_mean(h_theo)
        h_0 = meanGen.h_0
        avg_error = meanGen.avg_error[:,1].reshape((1,-1))
        hist_error = meanGen.error_history.reshape((1,-1))
        
        # update summary 
        summary["error_window"].append(avg_error)
        summary["error_hist"].append(hist_error)
        summary["mean"].append(h_0.mean()) 
        summary["var"].append(h_0.std())

        if (n % print_freq) == 0:
            print(f" n is :{n}")
    
    return summary

if __name__ == "__main__":
    # simple test of functions 
    params = DEFAULT_PARAMS_CONSTANT_MEAN
    res = main_one_over_n(params)
