# -*- coding: utf-8 -*-

import pdb
import numpy as np 

class AgentState(object):
    def __init__(self, s, x, q, idx_q = None, nu = 0):
        self.s = s
        self.x = x
        self.q = q
        
        self.idx_q = idx_q
        self.nu = nu

class ConstantAgent(object):
    
    def __init__(self, s_0, x_0, q_0, nb_iter = 100, time_step = 0.01, 
                 size_q = 80, q_max = 2, mu = 1, alpha = 0.1, 
                 var = 5, kappa = 0.1, phi = 1, **kwargs):
        
        self.s_0 = s_0
        self.x_0 = x_0
        self.q_0 = q_0
        self.mu = mu 
        self.alpha = alpha
        self.var = var
        self.kappa = kappa
        self.phi = phi
        
        self.nb_iter = nb_iter
        self.time_step = time_step
        
        self.size_q = size_q
        self.q_max = q_max
        self.q_min = -q_max
        self.step_q = (self.q_max - self.q_min)/self.size_q
        
        self.state = None
        self.next_state = None
    
    def initialize_variables(self):
        # Initialize state and next state
        idx_q = (self.q_0 - self.q_min) / self.step_q
        idx_q = int(round(min(max(idx_q, 0), self.size_q-1)))
        self.state = AgentState(self.s_0, self.x_0, self.q_0, idx_q)
        self.next_state = None
        
    def update(self, v_0, v_0_past, gamma= 0.2):
        # Initialize state variable
        v_0_cum = np.zeros((self.nb_iter + 1, self.size_q))
        self.initialize_variables()
        # Initialize random values
        z = np.random.normal(loc=0.0, scale=1.0, size=self.nb_iter)
        rnd_values = np.sqrt(self.time_step * self.var) * z
        # print( self.state.s,  self.state.x,  self.state.q,  self.state.idx_q)
        # pdb.set_trace()
        
        # Forward loop
        for i in range(self.nb_iter):
            # get next state
            self.next_state = self.getNextState(rnd_values[i], v_0_past[i+1,:])
            # print( self.next_state.s,  self.next_state.x,  self.next_state.q,  self.next_state.idx_q)
            # pdb.set_trace()

            # find optimal adjustment
            delta = self.find_optimal_adjust(i, v_0)
            v_0_cum[i, self.state.idx_q] += delta
            # pdb.set_trace()
            
            # update v_0
            v_0[i, self.state.idx_q] = v_0[i, self.state.idx_q] + gamma * v_0_cum[i, self.state.idx_q]# delta
            v_0_past[i, self.state.idx_q] = v_0_cum[i, self.state.idx_q] # delta    
            # pdb.set_trace()

            # Update current state values  
            self.state = self.next_state
            # self.next_state.nu
                   
        return v_0, v_0_past
    
    def getNextState(self, rnd_val, v_0_past):
        s_next = self.state.s + self.alpha * self.mu * self.time_step + rnd_val
        x_next = self.state.x - self.kappa * (self.state.nu*self.state.nu) * self.time_step\
                    - self.state.nu * self.state.s * self.time_step
        q_next = min(max(self.state.q + self.state.nu * self.time_step, self.q_min), self.q_max)
        idx_q = (self.state.q - self.q_min) / self.step_q
        idx_q = int(round(min(max(idx_q, 0), self.size_q-1)))
        
        weights = np.exp(5 * np.abs(v_0_past)) + 1e-4
        idx_q_next = np.random.choice(self.size_q, 1, p= weights/weights.sum())[0]
        nu_next = (self.q_min + idx_q_next * self.step_q - self.state.q) / self.time_step
        
        return AgentState(s_next, x_next, q_next, idx_q, nu_next)
        

    def find_optimal_adjust(self, i, v_0):
        # Next feasible controls
        q_val_min = max(self.q_min, self.q_min - self.state.q)
        q_val_max = min(self.q_max, self.q_max - self.state.q)
        q_consump_values = np.arange(q_val_min, q_val_max, self.step_q)
        q_consump_values = q_consump_values[q_consump_values != 0]

        q_next_values = (self.state.q + q_consump_values) # next inventories
        nu_next_values = q_consump_values / self.time_step # next trading speeds
        iq_next_values = (q_next_values - self.q_min) / self.step_q # next inventories indeces
        iq_next_values = np.rint(np.minimum(np.maximum(iq_next_values, 0), self.size_q-1)\
                                 ).astype(int)   
        
        # next wealths
        x_values_next = self.state.x - self.kappa * (nu_next_values*nu_next_values) * self.time_step\
                    - nu_next_values * self.state.s * self.time_step

        # next adjustments
        vect_values = (x_values_next - self.state.x) \
                + (q_next_values*self.next_state.s - self.state.q*self.state.s) \
                - self.phi * self.state.q * self.state.q * self.time_step \
                + v_0[i+1, iq_next_values] - v_0[i, self.state.idx_q]
        # pdb.set_trace()
        
        return vect_values.max() 
        
    def getLoss(self, v, v_theo):
        return np.linalg.norm(v - v_theo)
