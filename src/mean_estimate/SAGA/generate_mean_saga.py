# -*- coding: utf-8 -*-

import numpy as np 

from src.mean_estimate.SAGA.saga import SagaMean


class MeanGeneratorSaga(object):
    
    def __init__(self, s_val, nb_iter= 100, nb_episode = 100, 
                 window_size = 50, time_step = 0.01, 
                 mu = 1, alpha = 0.1, 
                 var = 5, gamma = 0.1, n_max = 1, prob_exp = 1,
                 print_metrics = True,
                 pctg_min = 0.1, **kwargs):
        
        self.nb_iter = nb_iter
        self.nb_episode = nb_episode
        self.window_size = window_size
        
        self.s_val = s_val
        self.time_step = time_step
        self.mu = mu 
        self.alpha = alpha
        self.var = var
        self.n_max = n_max
        self.prob_exp = prob_exp
        self.gamma = gamma
        
        self.pctg_min = pctg_min
        self.print_metrics = print_metrics
        
        # constant mean estimation
        self.sagaMean = SagaMean(self.s_val, self.nb_iter, self.time_step, 
                                 self.mu, self.alpha, self.var, self.n_max,
                                 self.prob_exp)
        
    def initialize_parameters(self):
        # initialize primary parameters
        self.h_0 = 100*np.ones(self.nb_iter)
        self.h_0_past = np.zeros((self.nb_iter + 1, self.n_max))
        self.nb_past = np.zeros(self.nb_iter + 1, dtype = int)
        
        # initialize tracking variables
        self.h_0_before = np.array(self.h_0)
        self.h_0_past_before = np.array(self.h_0_past)
        self.cnt_window = 0
        self.cnt_period = 0
        self.cnt_reset = 0
        self.avg_err_size = int(self.nb_episode//self.window_size)
        self.error_window = np.zeros(self.window_size)
        self.avg_error = np.zeros((self.avg_err_size,2))
        self.mean_window = np.zeros(self.window_size)
        self.avg_mean = np.zeros((self.avg_err_size,2))
        self.error_history = np.zeros(self.nb_episode)
        
    def get_mean(self, h_theo):
        # initialize parameters
        self.initialize_parameters()
        
        # loop over episodes
        for ep in range(self.nb_episode):
            # Update mean, error
            self.h_0, self.h_0_past = self.sagaMean.update(self.h_0, self.h_0_past,
                                                           self.nb_past,
                                                           gamma = self.gamma)
            error_val = self.sagaMean.getLoss(self.h_0, h_theo)

            # update tracking variables
            self.update_tracking_parameters(ep, error_val)
                    
    def update_tracking_parameters(self, ep, error_val):
        # update window variables
        self.error_window[self.cnt_window] = error_val
        self.mean_window[self.cnt_window] = np.linalg.norm(self.h_0)
        self.cnt_window += 1
        self.error_history[ep] = error_val
        
        # print summary
        self.print_summary(ep)
    
        # update gamma
        self.update_gamma(ep)
            
    def update_gamma(self, ep):
        if ((ep % self.window_size)==self.window_size-1 and (ep> 0)):
            # update avg variables
            self.avg_error[self.cnt_period] = (ep, self.error_window.mean())
            self.avg_mean[self.cnt_period] = (ep, self.mean_window.mean())
            
            # update gamma
            idx_bef = max(self.cnt_period-1,0)
            pctg_diff = (self.avg_mean[idx_bef,1] - self.avg_mean[self.cnt_period,1]) / self.avg_mean[idx_bef,1]
            if (pctg_diff <= self.pctg_min) and (self.cnt_period >= 1):
                # reset h_0, h_0_past
                self.h_0 = np.array(self.h_0_before)
                self.h_0_past = np.array(self.h_0_past_before)                
                if (self.cnt_reset >= 3):
                    self.gamma = max(self.gamma/2,0.01)
                    self.cnt_reset = 0
            else : 
                # update h_0_before, h_0_past_before
                self.h_0_before = np.array(self.h_0)
                self.h_0_past_before = np.array(self.h_0_past) 
            
            # reset variables
            self.error_window[:] = 0
            self.mean_window[:] = 0
            self.cnt_window = 0
            self.cnt_period += 1
            self.cnt_reset += 1
    
    def print_summary(self, ep):
        if self.print_metrics:
            if ((ep % self.window_size)==0 and (ep> 0)):
                print(f"Frequency is : {ep}")# use logger

if __name__ == "__main__":
    # Initialization parameters
    s_val = 0
    nb_iter= 25
    nb_episode = 1000
    window_size = 50
    time_step = 0.01 
    mu = 1
    alpha = 0.1
    n_max = 2
    prob_exp = 1
    var = 0.01
    gamma = 0.05
    print_metrics = True,
    pctg_min = 0.1
    h_theo = alpha * mu * time_step * np.ones(nb_iter) 
                 
    # Generate estimate
    meanGen = MeanGeneratorSaga(s_val, nb_iter, nb_episode, window_size,
                                time_step, mu, alpha, var, gamma,
                                n_max, prob_exp, print_metrics,
                                pctg_min)
    meanGen.get_mean(h_theo)
    