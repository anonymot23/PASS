# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 04:30:51 2023

@author: othma
"""

DEFAULT_PARAMS_CONSTANT_MEAN = {"s_val": 0,
                                "nbSimu": 200,
                                "nb_iter": 25,
                                "nb_episode": 1000,
                                "window_size": 50,
                                "time_step": 0.01,
                                "mu": 1,
                                "alpha": 0.1,
                                "var": 0.01,
                                "gamma": 0.05,
                                "pctg_min": 0.1,
                                "print_metrics": False,
                                "print_freq": 50
                                }

DEFAULT_PARAMS_PASS_MEAN = {"s_val": 0,
                            "nbSimu": 200,
                            "nb_iter": 25,
                            "nb_episode": 1000,
                            "window_size": 50,
                            "time_step": 0.01,
                            "mu": 1,
                            "alpha": 0.1,
                            "alpha_min": 1,
                            "alpha_max": 3,
                            "var": 0.01,
                            "gamma": 0.05,
                            "pctg_min": 0.1,
                            "print_metrics": False,
                            "print_freq": 50
                            }

DEFAULT_PARAMS_SAGA_MEAN = {"s_val": 0,
                            "nbSimu": 200,
                            "nb_iter": 25,
                            "nb_episode": 1000,
                            "window_size": 50,
                            "time_step": 0.01,
                            "mu": 1,
                            "alpha": 0.1,
                            "n_max": 2,
                            "prob_exp": 1,
                            "var": 0.01,
                            "gamma": 0.05,
                            "pctg_min": 0.1,
                            "print_metrics": False,
                            "print_freq": 50
                            }