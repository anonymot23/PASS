# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:36:51 2023

@author: othma
"""

from src.optimal_execution.SAGA.generate_v_saga import VGeneratorSaga
from src.utils.cj_theo_sol import CJTheoSol
from src.optimal_execution.parameters import DEFAULT_PARAMS_SAGA
 
def main_saga(params: dict = {}) -> None:
    # Initialization parameters
    if not(params):
        params = DEFAULT_PARAMS_SAGA
    
    s_0 = params["s_0"]
    x_0 = params["x_0"]
    q_0 = params["q_0"]
    nb_iter= params["nb_iter"]
    nbSimu = params["nbSimu"]
    nb_episode = params["nb_episode"]
    window_size = params["window_size"]
    time_step = params["time_step"] 
    size_q = params["size_q"]
    q_max = params["q_max"]
    T_max = params["T_max"]
    mu = params["mu"]
    alpha = params["alpha"]
    var = params["var"]
    kappa = params["kappa"]
    phi = params["phi"]
    n_max = params["n_max"]
    prob_exp = params["prob_exp"]
    A = params["A"]
    gamma = params["gamma"]
    print_metrics = params["print_metrics"]
    print_freq = params["print_freq"]
    pctg_min = params["pctg_min"]
    # Compute Theoretical values
    cjTheo = CJTheoSol(nb_iter, time_step, size_q, q_max, T_max, mu, alpha,
              kappa, phi, A)
    v_theo = cjTheo.get_v()
    
    # summary stats
    summary = {"error_window": [],
               "error_hist": [],
               "mean": [],
               "var": []}
                 
    # main routine
    for n in range(nbSimu):
        # generate forecast
        vGen = VGeneratorSaga(s_0, x_0, q_0, nb_iter, nb_episode, window_size,
                          time_step, size_q, q_max, mu, alpha,
                          var, kappa, phi, n_max, prob_exp,
                          A, gamma, print_metrics, pctg_min)
        vGen.get_v(v_theo)
        v_0 = vGen.v_0
        avg_error = vGen.avg_error[:,1].reshape((1,-1))
        hist_error = vGen.error_history.reshape((1,-1))
        
        # update summary 
        summary["error_window"].append(avg_error)
        summary["error_hist"].append(hist_error)
        summary["mean"].append(v_0.mean()) 
        summary["var"].append(v_0.std())

        if (n % print_freq) == 0:
            print(f" n is :{n}")
    
    return summary

if __name__ == "__main__":
    # simple test of functions 
    params = DEFAULT_PARAMS_SAGA
    res = main_saga(params)
