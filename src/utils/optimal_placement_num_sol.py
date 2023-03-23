# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd


class NumSol(object):
    
    def __init__(self, intensity_values, nb_iter= 100,  
                 size_q = 80, tol = 0.1, gain = 2, 
                 cost_out = -1, cost_stay = -0.5, **kwargs):
        
        self.intensity_values = intensity_values
        self.nb_iter = nb_iter
        
        self.size_q = size_q
        self.tol = tol
        
        self.gain = gain
        self.cost_out = cost_out
        self.cost_stay = cost_stay

    def get_prob(self, Q, z, size_q):
        """
        Takes the infinitesimal generator and return the probability 
        transition matrix and the average reward
        """
        size = size_q * size_q
        P = np.zeros((size, size))
        # Zero diagonal terms
        diagonal_indexes = np.arange(size)
        idx_true = Q[diagonal_indexes,diagonal_indexes] == 0
        idx_zero = diagonal_indexes[idx_true]
        P[idx_zero, idx_zero] = 1
        
        # Non zero diagonal terms
        idx_nzero = diagonal_indexes[~idx_true] 
        P[idx_nzero] = (-Q[idx_nzero]/Q[idx_nzero,idx_nzero][:,None])
        P[idx_nzero, idx_nzero] = 0
        z[idx_nzero] = -z[idx_nzero]/Q[idx_nzero,idx_nzero]
    
        return P, z
    
    def build_q_stay(self):
        """
        Build the infinitesimal generator  and reward for stay decision
        """
        tilde_size_q = self.size_q - 1
        size_Q_tilde = (tilde_size_q)*(tilde_size_q)
        Q_tilde = np.zeros((size_Q_tilde,size_Q_tilde))
        z_1 = np.zeros(size_Q_tilde)
        
        # Build Q transition matrix 
        for qsame in range(tilde_size_q):
            for bb_pos in range(qsame+1):
                CumIntens = 0.
                # Cancellation order bid side 
                if (qsame > 0) and (bb_pos == qsame) : ## the limit is not totally consumed  // No regeneration
                     CumIntens += self.intensity_values.loc[qsame + 1, 'Cancel'] 
                     Q_tilde[qsame*tilde_size_q+bb_pos][(qsame-1)*tilde_size_q+bb_pos-1] += self.intensity_values.loc[qsame + 1, 'Cancel'] 
                     z_1[qsame*tilde_size_q+bb_pos] += self.intensity_values.loc[qsame+1,'Cancel'] * self.reward_exec(qsame + 1, bb_pos + 1)
                elif (qsame > 0) and (bb_pos < qsame):
                     CumIntens +=  self.intensity_values.loc[qsame+1 ,'Cancel']
                     Q_tilde[qsame*tilde_size_q+bb_pos][(qsame-1)*tilde_size_q+bb_pos] += self.intensity_values.loc[qsame+1,'Cancel']
                     z_1[qsame*tilde_size_q+bb_pos] += self.intensity_values.loc[qsame+1,'Cancel'] * self.reward_exec(qsame + 1, bb_pos + 1)
    
                # Market order bid side
                if (bb_pos == 0) : ## the limit is not totally consumed  // No regeneration
                     CumIntens +=  self.intensity_values.loc[qsame+1 ,'Market'] 
                     z_1[qsame*tilde_size_q+bb_pos] += self.intensity_values.loc[qsame+1,'Market'] * self.reward_exec(qsame + 1, 0)
                elif (bb_pos > 0):
                     CumIntens +=  self.intensity_values.loc[qsame+1 ,'Market']
                     Q_tilde[qsame*tilde_size_q+bb_pos][(qsame-1)*tilde_size_q+bb_pos-1] += self.intensity_values.loc[qsame+1,'Market'] 
                     z_1[qsame*tilde_size_q+bb_pos] += self.intensity_values.loc[qsame+1,'Market'] * self.reward_exec(qsame + 1, bb_pos)
    
                                       
                # Insertion order bid side
                if (qsame < self.size_q -2) : ## when qsame = Qmax -1  no more order can be added to the bid limit
                     CumIntens += self.intensity_values.loc[qsame+1,'Limit'] # IntensVal['lambdaIns'][qsame*Qmax0+qopp]  
                     Q_tilde[qsame*tilde_size_q+bb_pos][(qsame+1)*tilde_size_q+bb_pos] += self.intensity_values.loc[qsame+1,'Limit']            
                     z_1[qsame*tilde_size_q+bb_pos] += self.intensity_values.loc[qsame+1,'Limit'] * self.reward_exec(qsame + 1, bb_pos + 1)
                
                
                # Nothing happen 
                Q_tilde[qsame*tilde_size_q+bb_pos][qsame*tilde_size_q+bb_pos] += (- CumIntens ) 
        
        Q_tilde, z_1 = self.get_prob(Q_tilde,np.array(z_1), tilde_size_q) 
        return [Q_tilde,z_1]
    
    def build_q_market(self):
        """
        Build the infinitesimal generator  and reward for market decision
        """
        tilde_size_q = self.size_q - 1
        size_Q_tilde = (tilde_size_q)*(tilde_size_q)
        z_1 = np.zeros(size_Q_tilde)
        
        ## Build Q transition matrix 
        for qsame in range(tilde_size_q) : # QSame Loop // qsame =0
            z_1[qsame*tilde_size_q : qsame*tilde_size_q + (qsame+1)] = self.reward_exec(qsame + 1, -1)
           
        return z_1
    
    def get_v(self):
        """
        Computation of the solution
        """
        # Build Q stay 
        Q_stay, z_stay = self.build_q_stay()
        # Build z_market
        z_mkt = self.build_q_market()
        
        tilde_size_q = self.size_q - 1
        u_1 = -5*np.ones(tilde_size_q*tilde_size_q)
        u_next_1 = -5*np.ones(tilde_size_q*tilde_size_q)
        for qsame in range(tilde_size_q): # qsame is the size
            u_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = 0
            u_next_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = 0
        index_next_1 = np.zeros(tilde_size_q*tilde_size_q)
        n = 0
        error = 2*self.tol+1
        while (n <= self.nb_iter) and (error > self.tol):
            # save old value ## debug
            u_1 = np.array(u_next_1)
            # stay value 
            stay_u = Q_stay.dot(u_next_1) + z_stay
            # market value
            market_u = z_mkt
            # compute optimal value
            u_next_1 = np.maximum(stay_u, market_u)
            # find optimal decision
            index_next_1 =  np.argmax([stay_u,market_u], axis = 0)
            # Update error
            error = np.sqrt(np.nan_to_num(np.abs(u_1 - u_next_1)).max())
            n += 1
    
        # process results
        stay_u = Q_stay.dot(u_next_1) + z_stay 
        market_u = z_mkt
        for qsame in range(tilde_size_q): # qsame is the size
            stay_u[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = np.nan
            market_u[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = np.nan
            index_next_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = -1
            u_next_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = np.nan
        
        # print results
        nb_col, nb_row = 6, tilde_size_q * tilde_size_q
        df = pd.DataFrame(np.zeros((nb_row, nb_col)), columns = ['BB size','BB pos','Limit','Market','Value opti','Decision'])
        df['BB size'] = np.repeat(np.arange(1, tilde_size_q + 1), tilde_size_q)
        df['BB pos'] = np.tile(np.arange(1, tilde_size_q + 1), tilde_size_q) 
        df['Limit'] = stay_u
        df['Market'] = market_u
        df['Value_opt'] = u_next_1
        df['Decision'] = index_next_1
        
        return df
    
    def reformat_sol(self, h):
        """
        Resize the result

        """
        h_0_theo = np.ones((self.size_q, self.size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
        h_0_theo[1:, 2:] = np.nan_to_num( h.reshape((self.size_q-1, self.size_q-1)) )
        # FINAL CONSTRAINT
        for qsame in range(self.size_q): # qsame is the size
            h_0_theo[qsame,qsame+2:] = np.nan
            h_0_theo[qsame,0] = self.reward_exec(qsame, -1)# market
            h_0_theo[qsame,1] = self.reward_exec(qsame, 0)# execution
        return h_0_theo

    def reward_exec(self, q, pos):
        # win when executed
        if pos ==  0: 
            return self.gain
        # cost of a market order
        elif pos ==  -1: 
            return self.cost_out
        # cost of waiting
        else : 
            return self.cost_stay