# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:36:51 2023

@author: othma
"""

from os.path import join
import pandas as pd
from src.optimal_placement.CONSTANT.generate_v_constant import VGeneratorConstant
from src.utils.optimal_placement_num_sol import NumSol
from src.optimal_placement.parameters import DEFAULT_PARAMS_CONSTANT
 
def main_constant(params: dict = {}) -> None:
    # Initialization parameters
    if not(params):
        params = DEFAULT_PARAMS_CONSTANT
    
    path = "..\..\..\data"
    filename = "Intens_val_qr.csv"
    Intens_val = pd.read_csv(join(path,filename), index_col = 0)
    Intens_val_bis = Intens_val[Intens_val['Spread'] == 1].groupby(['BB size']).agg({'Limit':'mean', 'Cancel': 'mean', 'Market': 'mean'}).loc[:10,:]
    Intens_val_bis.reset_index(inplace = True)
    Intens_val_bis.loc[0,['Cancel','Market']] = 0
    
    # Initialization parameters      
    q_0 = params["q_0"] 
    pos_0 = params["pos_0"] 
    q_0 = params["q_0"] 
    gain = params["gain"] 
    cost_out =  params["cost_out"] 
    cost_stay =  params["cost_stay"] 
    nb_iter=  params["nb_iter"] 
    nbSimu = params["nbSimu"]
    nb_episode =  params["nb_episode"] 
    window_size =  params["window_size"] 
    size_q =  params["size_q"] 
    q_max =  params["q_max"] 
    eta =  params["eta"] 
    gamma =  params["gamma"] 
    write_history =  params["write_history"] 
    print_metrics =  params["print_metrics"] 
    print_freq = params["print_freq"]
    pctg_min =  params["pctg_min"] 
    
    # Load and prepare  intensities
    path = params["path"]
    filename = params["filename"]
    intensity_values = pd.read_csv(join(path,filename), index_col = 0)
    intensity_values = intensity_values[Intens_val['Spread'] == 1].groupby(['BB size']).agg({'Limit':'mean', 'Cancel': 'mean', 'Market': 'mean'}).loc[:10,:]
    intensity_values.reset_index(inplace = True)
    intensity_values.loc[0,['Cancel','Market']] = 0 
 
    # Compute Theoretical values
    tol = 0.1
    size_q = Intens_val_bis.shape[0]
    nb_iter_scheme = 400
    num_sol = NumSol(intensity_values, nb_iter_scheme, \
                 size_q, tol, gain = 6, cost_out = -0.6, cost_stay = -0.2)
    df_bis = num_sol.get_v()
    h_theo = num_sol.reformat_sol(df_bis["Value_opt"].values)

    # summary stats
    summary = {"error_window": [],
               "error_hist": [],
               "mean": [],
               "var": []}
                 
    # main routine
    for n in range(nbSimu):
        # generate forecast
        vGen = VGeneratorConstant(q_0, pos_0, intensity_values,
                                  gain, cost_out, cost_stay,
                                  nb_iter, nb_episode, window_size,
                                  size_q, q_max, eta, 
                                  gamma, write_history, print_metrics,
                                  pctg_min)
        vGen.get_v(h_theo)
        h_0 = vGen.h_0
        avg_error = vGen.avg_error[:,1].reshape((1,-1))
        hist_error = vGen.error_history.reshape((1,-1))
        
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
    params = DEFAULT_PARAMS_CONSTANT
    res = main_constant(params)
