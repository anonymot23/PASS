# -*- coding: utf-8 -*-

from os.path import join
import numpy as np

from testing.optimal_placement.CONSTANT.main_constant import main_constant
from testing.optimal_placement.ONEOVERN.main_one_over_n import main_one_over_n
from testing.optimal_placement.SAGA.main_saga import main_saga
from testing.optimal_placement.PASS.main_pass import main_pass
from src.optimal_placement.parameters import DEFAULT_PARAMS_CONSTANT, \
    DEFAULT_PARAMS_SAGA, DEFAULT_PARAMS_PASS, IMAGES_FOLDER

import matplotlib.pyplot as plt

def main_compare(params: dict = {}) -> None:
    # Initialization parameters
    if not(params):
        params = {'constant': DEFAULT_PARAMS_CONSTANT,
                  'one_over_n': DEFAULT_PARAMS_CONSTANT,
                  'saga': DEFAULT_PARAMS_SAGA,
                  'pass': DEFAULT_PARAMS_PASS
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
    # Test functions 
    params = {'constant': DEFAULT_PARAMS_CONSTANT,
              'one_over_n': DEFAULT_PARAMS_CONSTANT,
              'saga': DEFAULT_PARAMS_SAGA,
              'pass': DEFAULT_PARAMS_PASS
              }
    res = main_compare(params)
    
    # plot values
    error1 = np.concatenate(res['one_over_n']['error_hist']).mean(axis=0)
    error2 = np.concatenate(res['constant']['error_hist']).mean(axis=0)
    error3 = np.concatenate(res['saga']['error_hist']).mean(axis=0)
    error4 = np.concatenate(res['pass']['error_hist']).mean(axis=0)
    index = np.arange(1, len(error1)+1)
    
    maxidx = 300
    plt.plot(index[:maxidx], error1[:maxidx], label='one_over_n')
    plt.plot(index[:maxidx], error2[:maxidx], label='constant')
    plt.plot(index[:maxidx], error3[:maxidx], label='saga')
    plt.plot(index[:maxidx], error4[:maxidx], label='pass')
    plt.grid()
    plt.legend()
    plt.savefig(join(IMAGES_FOLDER, "optimal_placement_comparison.png"),
                bbox_inches='tight')
    plt.show()