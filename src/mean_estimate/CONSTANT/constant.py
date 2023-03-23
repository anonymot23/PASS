# -*- coding: utf-8 -*-

import numpy as np 

class ConstantMean(object):
    
    def __init__(self, s_val, nb_iter= 100, time_step = 0.01, 
                 mu = 1, alpha = 0.1, var = 5, **kwargs):
        
        self.s_val = s_val
        self.nb_iter = nb_iter
        self.time_step = time_step
        self.mu = mu 
        self.alpha = alpha
        self.var = var
    
    def update(self, h_0, h_0_past, gamma= 0.2):
        s_val = self.s_val
        z = np.random.normal(loc=0.0, scale=1.0, size=self.nb_iter)
        rnd_values = np.sqrt(self.time_step * self.var) * z
        
        # Forward loop
        for i in range(self.nb_iter):
            s_val_next = s_val + self.alpha * self.mu * self.time_step + rnd_values[i]
            delta = (s_val_next - s_val - h_0[i])
            
            # Update h_0, h_0_past, s_val values
            h_0[i] = h_0[i] + gamma * delta
            h_0_past[i] = delta
            s_val = s_val_next
            
        return h_0, h_0_past
    
    def getLoss(self, h, h_theo):
        return np.linalg.norm(h - h_theo)