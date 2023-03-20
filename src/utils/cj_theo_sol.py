# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:54:13 2019

@author: othmane.mounjid
"""

## Import libraries
import numpy as np

class CJTheoSol(object):
    
    def __init__(self, nb_iter= 100, time_step = 0.01, 
                 size_q = 80, q_max = 2, T_max = 1, mu = 1, alpha = 0.1,
                 kappa = 0.1, phi = 1, A = 0.25, **kwargs):
        
        self.nb_iter = nb_iter
        
        self.time_step = time_step
        self.mu = mu 
        self.alpha = alpha
        self.kappa = kappa
        self.phi = phi
        self.A = A
        
        self.size_q = size_q
        self.q_max = q_max
        self.T_max = T_max
        self.q_min = -q_max
        self.step_q = (self.q_max - self.q_min)/self.size_q
     
    def get_v(self):
        # Compute Theoretical values
        v_theo = np.zeros((self.nb_iter + 1, self.size_q))
        h2 = self.get_h2()
        h1 = self.get_h1(h2)
        h0 = self.get_h0(h1)
        
        q_values = np.arange(self.q_min, self.q_max, self.step_q)
        for i in range(self.nb_iter + 1): # i = 0
            v_theo[i,:] = h0[i] + h1[i]*q_values - 0.5 * h2[i] * q_values * q_values
        
        return v_theo
                     
    def get_h2(self):
        a_1 = np.sqrt(4 * self.kappa * self.phi)
        a_2 = (-1/self.kappa) * a_1
        a_3 = -1/(2 * a_2 * self.kappa)
        a_4 = 1/(2 * self.A + a_1) - a_3
        time_values = np.arange(0, self.T_max + self.time_step, self.time_step)
        h2 = 1/(a_3 + a_4*np.exp(a_2 * (self.T_max  - time_values))) - a_1
        return h2
    
    
    def get_h1(self, h2):
        h1 = np.zeros(self.nb_iter + 1)
        
        # Terminal condition
        h1[-1] = 0
        
        # Backward loop 
        for i in range(self.nb_iter): # i = 0
            h1[self.nb_iter - (i+1)] = h1[self.nb_iter - i] - ((h2[self.nb_iter - i]/(2 * self.kappa)) * h1[self.nb_iter - i] - self.alpha * self.mu) *  self.time_step
    
        return h1
    
    def get_h0(self, h1):
        h0 = np.zeros(self.nb_iter + 1)
        
        # Terminal condition
        h0[-1] = 0
        
        # Backward loop 
        for i in range(self.nb_iter): # i = 0
            h0[self.nb_iter - (i+1)] = h0[self.nb_iter - i] - (-(h1[self.nb_iter-i] * h1[self.nb_iter-i])/( 4 * self.kappa)) * self.time_step
    
        return h0
    