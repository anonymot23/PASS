# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 04:30:51 2023

@author: othma
"""

DEFAULT_PARAMS_CONSTANT = { "q_0": 5,
                            "pos_0": 0,
                            "gain": 2,
                            "cost_out": -1,
                            "cost_stay": -0.5,
                            "nbSimu": 200,
                            "nb_iter": 25,
                            "nb_episode": 1000,
                            "window_size": int(50),
                            "size_q": 80,
                            "q_max": 2,
                            "eta": 1,
                            "gamma": 0.05,
                            "write_history": False,
                            "pctg_min": 0.1,
                            "print_metrics": False,
                            "print_freq": 50,
                            "path": "..\..\..\data",
                            "filename": "Intens_val_qr.csv"
                            }

DEFAULT_PARAMS_PASS = { "q_0": 5,
                        "pos_0": 0,
                        "gain": 2,
                        "cost_out": -1,
                        "cost_stay": -0.5,
                        "alpha_min": 1,
                        "alpha_max": 3,
                        "r": 0.5,
                        "nbSimu": 200,
                        "nb_iter": 25,
                        "nb_episode": 1000,
                        "window_size": int(50),
                        "size_q": 80,
                        "q_max": 2,
                        "eta": 1,
                        "gamma": 0.05,
                        "write_history": False,
                        "pctg_min": 0.1,
                        "print_metrics": False,
                        "print_freq": 50,
                        "path": "..\..\..\data",
                        "filename": "Intens_val_qr.csv"
                       }
    
DEFAULT_PARAMS_SAGA = { "q_0": 5,
                        "pos_0": 0,
                        "gain": 2,
                        "cost_out": -1,
                        "cost_stay": -0.5,
                        "n_max": 1,
                        "prob_exp": 1,
                        "nbSimu": 200,
                        "nb_iter": 25,
                        "nb_episode": 1000,
                        "window_size": int(50),
                        "size_q": 80,
                        "q_max": 2,
                        "eta": 1,
                        "gamma": 0.05,
                        "write_history": False,
                        "pctg_min": 0.1,
                        "print_metrics": False,
                        "print_freq": 50,
                        "path": "..\..\..\data",
                        "filename": "Intens_val_qr.csv"
                       }