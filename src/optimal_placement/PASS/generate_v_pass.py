# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:51:37 2019

@author: othmane.mounjid
"""

import numpy as np 

from src.optimal_placement.PASS.pass_ import PassAgent

class VGeneratorPass(object):

    def __init__(self, q_0, pos_0, intensity_values,
                 gain = 2, cost_out = -1, cost_stay = -0.5,
                 alpha_min = 1, alpha_max = 3, r = 0.5,
                 nb_iter= 100, nb_episode = 100, window_size = 50,
                 size_q = 80, q_max = 2, eta = 1, 
                 gamma = 0.1, write_history = False, print_metrics = True,
                 pctg_min = 0.1, **kwargs):
        
        self.nb_iter = nb_iter
        self.nb_episode = nb_episode
        self.window_size = window_size
        
        self.q_0 = q_0
        self.pos_0 = pos_0
        self.intensity_values = intensity_values
        self.gamma = gamma
        
        self.gain = gain
        self.cost_out = cost_out 
        self.cost_stay = cost_stay
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.r = r
        
        self.size_q = size_q
        self.q_max = q_max
        self.eta = eta

        self.pctg_min = pctg_min
        self.write_history = write_history
        self.print_metrics = print_metrics
        
        # agent
        self.agent = PassAgent(self.q_0, self.pos_0, self.intensity_values,
                               self.gain, self.cost_out, self.cost_stay,
                               self.alpha_min, self.alpha_max, self.r,
                               self.nb_iter, self.size_q, self.q_max, 
                               self.eta, self.write_history)
        
        
    def initialize_parameters(self):
        # initialize primary parameters
        self.h_0 = np.ones((self.size_q, self.size_q + 1))
        self.h_0_stay = np.ones((self.size_q, self.size_q + 1))
        self.h_0_mkt = np.ones((self.size_q, self.size_q + 1))
        # add final constraint
        for q in range(self.size_q):
            self.h_0[q, q + 2 :] = np.nan
            self.h_0_stay[q, q + 2 :] = np.nan
            self.h_0_mkt[q, q + 2 :] = np.nan
        self.h_0_past = np.zeros((self.size_q, self.size_q + 1, 3))
        self.nb_past = np.zeros((self.size_q, self.size_q + 1),\
                                dtype = int)
        
        # initialize tracking variables
        self.h_0_before = np.array(self.h_0)
        self.h_0_stay_before = np.array(self.h_0_stay)
        self.h_0_mkt_before = np.array(self.h_0_mkt)
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

        
    def get_v(self, h_theo):
        # initialize parameters
        self.initialize_parameters()
        
        # loop over episodes
        for ep in range(self.nb_episode):
            # Update h_0
            self.h_0_mkt, self.h_0_stay, self.h_0, self.h_0_past  = self.agent.update(self.h_0_mkt, self.h_0_stay,
                                                            self.h_0, 
                                                            self.h_0_past,
                                                            self.nb_past,
                                                            gamma= self.gamma)
            error_val = self.agent.getLoss(self.h_0, h_theo)

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
            pctg_diff = (self.avg_error[idx_bef,1] - self.avg_error[self.cnt_period,1]) / self.avg_error[idx_bef,1]
            if (pctg_diff <= self.pctg_min) and (self.cnt_period >= 1)\
                and (self.cnt_reset >= 3):
                # reset h_0(s)
                self.h_0 = np.array(self.h_0_before)
                self.h_0_stay = np.array(self.h_0_stay_before)
                self.h_0_mkt = np.array(self.h_0_mkt_before) 
                self.h_0_past = np.array(self.h_0_past_before)                
            else : 
                # update h_0(s)_before
                self.h_0_before = np.array(self.h_0)
                self.h_0_stay_before = np.array(self.h_0_stay)
                self.h_0_mkt_before = np.array(self.h_0_mkt) 
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
    from os.path import join
    import pandas as pd
    import src.optimal_placement.rLAlgorithms.solTheo as sol_theo
    
    def reward_exec(qsame, bb_pos, gain = 2, cost_out = -1, cost_stay = -0.5):
        if bb_pos ==  0: ## win if execution
            return gain
        elif bb_pos ==  -1: ## cost of a market order
            return cost_out
        else : ## cost of waiting
            return cost_stay
    
    path = "..\..\..\data"
    filename = "Intens_val_qr.csv"
    Intens_val = pd.read_csv(join(path,filename), index_col = 0)
    Intens_val_bis = Intens_val[Intens_val['Spread'] == 1].groupby(['BB size']).agg({'Limit':'mean', 'Cancel': 'mean', 'Market': 'mean'}).loc[:10,:]
    Intens_val_bis.reset_index(inplace = True)
    Intens_val_bis.loc[0,['Cancel','Market']] = 0
    
    # Initialization parameters      
    q_0 = 5
    pos_0 = 0
    intensity_values = Intens_val_bis
    q_0 = 1 
    gain = 2
    cost_out = -1
    cost_stay = -0.5,
    alpha_min = 1
    alpha_max = 3
    r = 0.5
    nb_iter= 100
    nb_episode = int(100)
    window_size = 50
    size_q = 80
    q_max = 2
    eta = 1
    gamma = 0.05
    write_history = False
    print_metrics = True
    pctg_min = 0.1
 
    # Compute Theoretical values
    tol = 0.1
    size_q = Intens_val_bis.shape[0]
    nb_iter_scheme = 400
    reward_exec_1 = lambda qsame, bb_pos: reward_exec(qsame, bb_pos, gain = 6, cost_out = -0.6, cost_stay = -0.2)
    df_bis = sol_theo.Sol_num_scheme(nb_iter_scheme,size_q,Intens_val_bis,tol = tol,reward_exec_1 = reward_exec_1)
    h_theo = sol_theo.Read_h_0_theo(df_bis["Value_opt"].values, size_q, reward_exec_1)

    
    # Generate forecast
    vGen = VGeneratorPass(q_0, pos_0, intensity_values,
                        gain, cost_out, cost_stay,
                        alpha_min, alpha_max, r, nb_iter, 
                        nb_episode, window_size,
                        size_q, q_max, eta, 
                        gamma, write_history, print_metrics,
                        pctg_min)
    vGen.get_v(h_theo)