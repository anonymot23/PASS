# -*- coding: utf-8 -*-

DATA_FOLDER = "..\..\..\data"

INTENSITY_FILENAME = "Intens_val_qr.csv"
    
IMAGES_FOLDER = "..\..\..\outputs\images"

DEFAULT_PARAMS_CONSTANT = { "q_0": 2,
                            "pos_0": 1,
                            "gain": 6,
                            "cost_out": -0.6,
                            "cost_stay": -0.2,
                            "nbSimu": 1,
                            "nb_iter": 100,
                            "nb_episode": 500,
                            "window_size": int(50),
                            "size_q": 80,
                            "q_max": 2,
                            "eta": 1,
                            "gamma": 0.1,
                            "write_history": True,
                            "pctg_min": 0.1,
                            "print_metrics": False,
                            "print_freq": 50,
                            "path": "..\..\..\data",
                            "filename": "Intens_val_qr.csv"
                            }
    
DEFAULT_PARAMS_PASS = { "q_0": 2,
                        "pos_0": 1,
                        "gain": 6,
                        "cost_out": -0.6,
                        "cost_stay": -0.2,
                        "alpha_min": 1,
                        "alpha_max": 4,
                        "r": 2/3,
                        "nbSimu": 1,
                        "nb_iter": 100,
                        "nb_episode": 500,
                        "window_size": int(50),
                        "size_q": 80,
                        "q_max": 2,
                        "eta": 1,
                        "gamma": 0.1,
                        "write_history": True,
                        "pctg_min": 0.1,
                        "print_metrics": False,
                        "print_freq": 50,
                        "path": "..\..\..\data",
                        "filename": "Intens_val_qr.csv"
                        }

DEFAULT_PARAMS_SAGA = { "q_0": 2,
                        "pos_0": 1,
                        "gain": 6,
                        "cost_out": -0.6,
                        "cost_stay": -0.2,
                        "n_max": 1,
                        "prob_exp": 1,
                        "nbSimu": 1,
                        "nb_iter": 100,
                        "nb_episode": 500,
                        "window_size": int(50),
                        "size_q": 80,
                        "q_max": 2,
                        "eta": 1,
                        "gamma": 0.1,
                        "write_history": True,
                        "pctg_min": 0.1,
                        "print_metrics": False,
                        "print_freq": 50,
                        "path": "..\..\..\data",
                        "filename": "Intens_val_qr.csv"
                        }