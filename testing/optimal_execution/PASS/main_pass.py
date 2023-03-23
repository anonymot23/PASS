# -*- coding: utf-8 -*-

from src.optimal_execution.PASS.generate_v_pass import VGeneratorPass
from src.utils.cj_theo_sol import CJTheoSol
from src.optimal_execution.parameters import DEFAULT_PARAMS_PASS
 
def main_pass(params: dict = {}) -> None:
    # Initialization parameters
    if not(params):
        params = DEFAULT_PARAMS_PASS

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
    alpha_min = params["alpha_min"]
    alpha_max = params["alpha_max"]
    r = params["r"]
    var = params["var"]
    kappa = params["kappa"]
    phi = params["phi"]
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
        vGen = VGeneratorPass(s_0, x_0, q_0, nb_iter, nb_episode, window_size,
                              time_step, size_q, q_max, mu, alpha,
                              alpha_min, alpha_max, r, var, kappa, phi,
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
    # Test functions 
    params = DEFAULT_PARAMS_PASS
    res = main_pass(params)
