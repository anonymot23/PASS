# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:36:51 2023

@author: othma
"""

import numpy as np

from mean_estimate.CONSTANT.main_constant import main_constant
from mean_estimate.ONEOVERN.main_one_over_n import main_one_over_n
from mean_estimate.SAGA.main_saga import main_saga
from mean_estimate.PASS.main_pass import main_pass
from src.mean_estimate.parameters import DEFAULT_PARAMS_CONSTANT_MEAN, \
    DEFAULT_PARAMS_SAGA_MEAN, DEFAULT_PARAMS_PASS_MEAN

import matplotlib.pyplot as plt

def main_compare(params: dict = {}) -> None:
    # Initialization parameters
    if not(params):
        params = {'cste': DEFAULT_PARAMS_CONSTANT_MEAN,
                  'one_over_n': DEFAULT_PARAMS_CONSTANT_MEAN,
                  'saga': DEFAULT_PARAMS_SAGA_MEAN,
                  'pass': DEFAULT_PARAMS_PASS_MEAN
                 }
    
    summary = dict()
    if 'constant' in params:
        summary['constant'] = main_constant(params['constant'])
    if 'one_over_n' in params:
        summary['one_over_n'] = main_one_over_n(params['one_over_n'])
    if 'saga' in params:
        summary['saga'] = main_saga(params['saga'])
    if 'pass' in params:
        summary['pass'] = main_pass(params['pass'])
    
    return summary

if __name__ == "__main__":
    # simple test of functions 
    params = {'constant': DEFAULT_PARAMS_CONSTANT_MEAN,
              'one_over_n': DEFAULT_PARAMS_CONSTANT_MEAN,
              'saga': DEFAULT_PARAMS_SAGA_MEAN,
              'pass': DEFAULT_PARAMS_PASS_MEAN
             }
    res = main_compare(params)
    
    # plot values
    error1 = np.concatenate(res['one_over_n']['error_hist']).mean(axis=0)
    error2 = np.concatenate(res['constant']['error_hist']).mean(axis=0)
    error3 = np.concatenate(res['saga']['error_hist']).mean(axis=0)
    error4 = np.concatenate(res['pass']['error_hist']).mean(axis=0)
    index = np.arange(1, len(error1)+1)
    
    plt.plot(index, error1, label='one_over_n')
    plt.plot(index, error2, label='constant')
    plt.plot(index, error3, label='saga')
    plt.plot(index, error4, label='pass')
    plt.grid()
    plt.legend()
    plt.show()
