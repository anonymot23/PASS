# -*- coding: utf-8 -*-

DATA_FOLDER = "..\..\..\data"

INTENSITY_FILENAME = "Intens_val_qr.csv"
    
IMAGES_FOLDER = "..\..\..\outputs\images"


DEFAULT_PARAMS_CONSTANT = { "q_0": 2,
                            "pos_0": 1,
                            "gain": 2,
                            "cost_out": -1,
                            "cost_stay": -0.5,
                            "nbSimu": 2,
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
                            "gain": 2,
                            "cost_out": -1,
                            "cost_stay": -0.5,
                            "alpha_min": 1,
                            "alpha_max": 3,
                            "r": 0.5,
                            "nbSimu": 2,
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
                            "gain": 2,
                            "cost_out": -1,
                            "cost_stay": -0.5,
                            "n_max": 1,
                            "prob_exp": 1,
                            "nbSimu": 2,
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