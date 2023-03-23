# -*- coding: utf-8 -*-

import numpy as np 

from src.optimal_placement.SAGA.saga import SagaAgent
from src.optimal_placement.CONSTANT.constant import BookState

class VGeneratorSaga(object):

    def __init__(self, q_0, pos_0, intensity_values,
                 gain = 2, cost_out = -1, cost_stay = -0.5,
                 n_max = 1, prob_exp = 1, nb_iter= 100,
                 nb_episode = 100, window_size = 50,
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
        self.n_max = n_max
        self.prob_exp = prob_exp
        
        self.size_q = size_q
        self.q_max = q_max
        self.eta = eta

        self.pctg_min = pctg_min
        self.write_history = write_history
        self.print_metrics = print_metrics
        
        # agent
        self.agent = SagaAgent(self.q_0, self.pos_0, self.intensity_values,
                               self.gain, self.cost_out, self.cost_stay,
                               self.n_max, self.prob_exp, self.nb_iter,
                               self.size_q, self.q_max, 
                               self.eta, self.write_history)
        
        
    def initialize_parameters(self):
        # initialize primary parameters
        self.h_0 = 5*np.ones((self.size_q, self.size_q + 1))
        self.h_0_stay = np.ones((self.size_q, self.size_q + 1))
        self.h_0_mkt = np.ones((self.size_q, self.size_q + 1))
        # add final constraint
        for q in range(self.size_q):
            self.h_0[q, q + 2 :] = np.nan
            self.h_0_stay[q, q + 2 :] = np.nan
            self.h_0_mkt[q, q + 2 :] = np.nan
            self.h_0[q, 0] = self.get_reward(q, -1)# market
            self.h_0[q, 1] = self.get_reward(q, 0)# execution
        self.h_0_past = np.zeros((self.size_q, self.size_q + 1, 3,\
                                  self.n_max))
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

    def get_reward(self, q, pos):
        state = BookState(q, pos)
        return self.agent.get_reward(state)
    
    def print_summary(self, ep):
        if self.print_metrics:
            if ((ep % self.window_size)==0 and (ep> 0)):
                print(f"Frequency is : {ep}")# use logger

if __name__ == "__main__":
    from os.path import join
    import pandas as pd
    from src.utils.optimal_placement_num_sol import NumSol
    from src.optimal_placement.parameters import DATA_FOLDER, INTENSITY_FILENAME
    
    Intens_val = pd.read_csv(join(DATA_FOLDER, INTENSITY_FILENAME), index_col = 0)
    Intens_val_bis = Intens_val[Intens_val['Spread'] == 1].groupby(['BB size']).agg({'Limit':'mean', 'Cancel': 'mean', 'Market': 'mean'}).loc[:10,:]
    Intens_val_bis.reset_index(inplace = True)
    Intens_val_bis.loc[0,['Cancel','Market']] = 0
    
    # Initialization parameters      
    q_0 = 2
    pos_0 = 1
    intensity_values = Intens_val_bis
    gain = 6
    cost_out = -0.6
    cost_stay = -0.2
    n_max = 1
    prob_exp = 1
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
    num_sol = NumSol(intensity_values, nb_iter_scheme, \
                 size_q, tol, gain = 6, cost_out = -0.6, cost_stay = -0.2)
    df_bis = num_sol.get_v()
    h_theo = num_sol.reformat_sol(df_bis["Value_opt"].values)
    
    # Generate forecast
    vGen = VGeneratorSaga(q_0, pos_0, intensity_values,
                        gain, cost_out, cost_stay,
                        n_max, prob_exp, nb_iter, 
                        nb_episode, window_size,
                        size_q, q_max, eta, 
                        gamma, write_history, print_metrics,
                        pctg_min)
    vGen.get_v(h_theo)