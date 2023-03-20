# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:51:37 2019

@author: othmane.mounjid
"""

import numpy as np 

from src.optimal_execution.CONSTANT.constant import ConstantAgent

class VGeneratorOneOverN(object):
    
    def __init__(self, s_0, x_0, q_0, nb_iter= 100, nb_episode = 100, 
                 window_size = 50, time_step = 0.01, 
                 size_q = 80, q_max = 2, mu = 1, alpha = 0.1,
                 var = 5, kappa = 0.1, phi = 1, A = 0.25, gamma = 0.1,
                 print_metrics = True, pctg_min = 0.1, **kwargs):
        
        self.nb_iter = nb_iter
        self.nb_episode = nb_episode
        self.window_size = window_size
        
        self.s_0 = s_0
        self.x_0 = x_0
        self.q_0 = q_0
        self.time_step = time_step
        self.mu = mu 
        self.alpha = alpha
        self.var = var
        self.kappa = kappa
        self.phi = phi
        self.A = A
        self.gamma = gamma
        
        self.size_q = size_q
        self.q_max = q_max

        self.pctg_min = pctg_min
        self.print_metrics = print_metrics
        
        # constant mean estimation
        self.constAgent = ConstantAgent(self.s_0, self.x_0, self.q_0,
                                       self.nb_iter, self.time_step, 
                                       self.size_q, self.q_max, self.mu,
                                       self.alpha, self.var, self.kappa,
                                       self.phi)
        
        
    def initialize_parameters(self):
        # initialize primary parameters
        self.v_0 = np.ones((self.nb_iter + 1, self.size_q))
        q_values = np.arange(self.constAgent.q_min, self.constAgent.q_max,
                             self.constAgent.step_q)
        self.v_0[-1,:] = - self.A * q_values * q_values
        self.v_0_past = np.zeros((self.nb_iter + 1, self.size_q))
        
        # initialize tracking variables
        self.v_0_before = np.array(self.v_0)
        self.v_0_past_before = np.array(self.v_0_past)
        self.cnt_window = 0
        self.cnt_period = 0
        self.cnt_reset = 0
        self.avg_err_size = int(self.nb_episode//self.window_size)
        self.error_window = np.zeros(self.window_size)
        self.avg_error = np.zeros((self.avg_err_size,2))
        self.mean_window = np.zeros(self.window_size)
        self.avg_mean = np.zeros((self.avg_err_size,2))
        self.error_history = np.zeros(self.nb_episode)

        
    def get_v(self, v_theo):
        # initialize parameters
        self.initialize_parameters()
        
        # loop over episodes
        for ep in range(self.nb_episode):
            # Update mean, error
            gamma_inner = self.update_gamma_inner(ep)
            self.v_0, self.v_0_past = self.constAgent.update(self.v_0, self.v_0_past,
                                                            gamma= gamma_inner)
            error_val = self.constAgent.getLoss(self.v_0, v_theo)

            # update tracking variables
            self.update_tracking_parameters(ep, error_val)
                    
    def update_tracking_parameters(self, ep, error_val):
        # update window variables
        self.error_window[self.cnt_window] = error_val
        self.mean_window[self.cnt_window] = np.linalg.norm(self.v_0)
        self.cnt_window += 1
        self.error_history[ep] = error_val
        
        # print summary
        self.print_summary(ep)
    
        # update gamma
        self.update_gamma(ep)
    
    def update_gamma_inner(self, ep):
        n = max((1 + (ep//40))*0.2, 1)
        return self.gamma/n
    
    def update_gamma(self, ep):
        if ((ep % self.window_size)==self.window_size-1 and (ep> 0)):
            # update avg variables
            self.avg_error[self.cnt_period] = (ep, self.error_window.mean())
            self.avg_mean[self.cnt_period] = (ep, self.mean_window.mean())
            
            # update gamma
            idx_bef = max(self.cnt_period-1,0)
            pctg_diff = (self.avg_error[idx_bef,1] - self.avg_error[self.cnt_period,1]) / self.avg_error[idx_bef,1]
            if (pctg_diff <= self.pctg_min) and (self.cnt_period >= 1):
                # reset v_0, v_0_past
                self.v_0 = np.array(self.v_0_before)
                self.v_0_past = np.array(self.v_0_past_before)                
                if (self.cnt_reset >= 5):
                    self.gamma = max(self.gamma - 0.01, 0.01)
                    self.cnt_reset = 0
                self.cnt_reset += 1
            else : 
                # update v_0_before, v_0_past_before
                self.v_0_before = np.array(self.v_0)
                self.v_0_past_before = np.array(self.v_0_past) 
            
            # reset variables
            self.error_window[:] = 0
            self.mean_window[:] = 0
            self.cnt_window = 0
            self.cnt_period += 1
    
    def print_summary(self, ep):
        if self.print_metrics:
            if ((ep % self.window_size)==0 and (ep> 0)):
                print(f"Frequency is : {ep}")# use logger

if __name__ == "__main__":
    from src.utils.cj_theo_sol import CJTheoSol 
    
    # Initialization parameters
    s_0 = 5
    x_0 = 0
    q_0 = 1 
    nb_iter= 25
    nb_episode = int(1000)
    window_size = 1000
    time_step = 0.01 
    size_q = 80
    q_max = 2
    T_max = 1
    mu = 1
    alpha = 0.1
    var = 0.01
    kappa = 0.1
    phi = 1
    A = 0.25
    gamma = 0.05
    print_metrics = True
    pctg_min = 0.1
 
    # Compute Theoretical values
    cjTheo = CJTheoSol(nb_iter, time_step, size_q, q_max, T_max, mu, alpha,
              kappa, phi, A)
    v_theo = cjTheo.get_v()
    
    # Generate forecast
    vGen = VGeneratorOneOverN(s_0, x_0, q_0, nb_iter, nb_episode, window_size,
                              time_step, size_q, q_max, mu, alpha,
                              var, kappa, phi, A, gamma, 
                              print_metrics, pctg_min)
    vGen.get_v(v_theo)
    