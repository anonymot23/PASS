# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:51:37 2019

@author: othmane.mounjid
"""

import numpy as np 

class PassMean(object):
    
    def __init__(self, s_val, nb_iter= 100, time_step = 0.01, 
                 mu = 1, alpha = 0.1, var = 5,
                 alpha_min = 1, alpha_max = 3,
                 **kwargs):
        
        self.s_val = s_val
        self.nb_iter = nb_iter
        self.time_step = time_step
        self.mu = mu 
        self.alpha = alpha
        self.alpha_curr = alpha_min
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.var = var
    
    def update(self, h_0, h_0_past, nb_past, gamma = 0.2):
        s_val = self.s_val
        z = np.random.normal(loc=0.0, scale=1.0, size=self.nb_iter)
        rnd_values = np.sqrt(self.time_step * self.var) * z
        
        # Forward loop
        for i in range(self.nb_iter):
            s_val_next = s_val + self.alpha * self.mu * self.time_step + rnd_values[i]
            delta = (s_val_next - s_val - h_0[i])
            
            # Update h_0, h_0_past, s_val values
            if (nb_past[i] == 0):# state never visited
                h_0[i] = h_0[i] + gamma * delta
            elif ( delta*h_0_past[i] >0 ):
                self.alpha_curr = min(self.alpha_curr +1, self.alpha_max)
                h_0[i] = h_0[i] + self.alpha_curr * gamma * delta
            elif ( delta*h_0_past[i] <=0 ):
                self.alpha_curr = max(self.alpha_curr -1, self.alpha_min)
                h_0[i] = h_0[i] + self.alpha_curr * gamma * delta
                
            h_0_past[i] = delta
            nb_past[i] += 1 
            s_val = s_val_next
            
        return h_0, h_0_past
    
    def reset_alpha_curr(self):
        self.alpha_curr = self.alpha_min
    
    def getLoss(self, h, h_theo):
        return np.linalg.norm(h - h_theo)