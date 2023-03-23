# -*- coding: utf-8 -*-

import numpy as np 

class SagaMean(object):
    
    def __init__(self, s_val, nb_iter= 100, time_step = 0.01, 
                 mu = 1, alpha = 0.1, var = 5,
                 n_max = 1, prob_exp = 1, **kwargs):
        
        self.s_val = s_val
        self.nb_iter = nb_iter
        self.time_step = time_step
        self.mu = mu 
        self.alpha = alpha
        self.var = var
        
        self.n_max = n_max
        self.prob_exp = prob_exp
        self.exp_mean = self.prob_exp ** (np.arange(self.n_max, 0, -1))
        
    def update(self, h_0, h_0_past, nb_past, gamma= 0.2):
        s_val = self.s_val
        z = np.random.normal(loc=0.0, scale=1.0, size=self.nb_iter)
        rnd_values = np.sqrt(self.time_step * self.var) * z
        
        # Forward loop
        for i in range(self.nb_iter):
            s_val_next = s_val + self.alpha * self.mu * self.time_step + rnd_values[i]
            delta = (s_val_next - s_val - h_0[i])
            j = nb_past[i]
            
            # Update h_0, h_0_past, s_val values
            if (j == 0):# state never visited
                h_0[i] = h_0[i] + gamma * delta
                h_0_past[i, j] = delta 
            elif (j < self.n_max-1):
                r = np.random.randint(0,j)
                avg_val = (h_0_past[i,:j] * self.exp_mean[:j]).sum() / self.exp_mean[:j].sum()
                h_0[i] = h_0[i] + gamma * (delta - h_0_past[i,r]  + avg_val)
                h_0_past[i, j] = delta
            else :
                r = np.random.randint(0, self.n_max)
                avg_val = (h_0_past[i,:] * self.exp_mean[:]).sum() / self.exp_mean[:].sum()
                h_0[i] = h_0[i] + gamma * (delta - h_0_past[i,r]  + avg_val)
                h_0_past[i,r] = delta 
                
            nb_past[i] += 1 
            s_val = s_val_next
            
        return h_0, h_0_past
    
    def getLoss(self, h, h_theo):
        return np.linalg.norm(h - h_theo)