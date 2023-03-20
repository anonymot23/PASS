# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:36:51 2023

@author: othma
"""

import numpy as np

from src.mean_estimate.CONSTANT.generate_mean_constant import MeanGeneratorConstant
from src.mean_estimate.parameters import DEFAULT_PARAMS_CONSTANT_MEAN
 
def main_constant(params: dict = {}) -> None:
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
    summary = {"error": np.zeros(nbSimu),
               "mean": np.zeros(nbSimu),
               "var": np.zeros(nbSimu)}
                 
    # main routine
    for n in range(nbSimu):
        # generate forecast
        meanGenCste = MeanGeneratorConstant(s_val, nb_iter, nb_episode, window_size,
                                            time_step, mu, alpha, var, gamma, 
                                            print_metrics, pctg_min)
        meanGenCste.get_mean(h_theo)
        h_0 = meanGenCste.h_0
        avg_error = meanGenCste.avg_error[-1]
        error_estim = avg_error[avg_error!=0][-1]
        
        # update summary 
        summary["error"][n] = error_estim
        summary["mean"][n] = h_0.mean()
        summary["var"][n] = h_0.std()

        if (n % print_freq) == 0:
            print(f" n is :{n}")
    
    return summary

if __name__ == "__main__":
    # simple test of functions 
    params = DEFAULT_PARAMS_CONSTANT_MEAN
    res = main_constant(params)
