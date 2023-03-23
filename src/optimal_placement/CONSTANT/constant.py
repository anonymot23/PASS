# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd

class BookState(object):
    def __init__(self, q, pos):
        self.q = q
        self.pos = pos

class ConstantAgent(object):
    def __init__(self, q_0, pos_0, intensity_values, 
                 gain = 2, cost_out = -1, cost_stay = -0.5,
                 nb_iter = 100, size_q = 80, q_max = 2,
                 eta = 1, write_history = False,
                 **kwargs):
        
        self.q_0 = q_0
        self.pos_0 = pos_0
        
        self.intensity_values = intensity_values
        self.columns_intensity = ['Limit', 'Cancel', 'Market']
        
        self.gain = gain
        self.cost_out = cost_out
        self.cost_stay = cost_stay
        
        self.nb_iter = nb_iter
        
        self.size_q = size_q
        self.q_max = q_max
        self.q_min = -q_max
        self.step_q = (self.q_max - self.q_min)/self.size_q
        self.eta = eta
        
        self.state = None
        self.next_state = None
        
        self.write_history = write_history
        self.book_history = None
    
    def initialize_variables(self):
        # Initialize state and next state
        self.state = BookState(self.q_0, self.pos_0)
        self.next_state = None
        self.book_history = self.initialize_book()
    
    def initialize_book(self):
        if self.write_history:
            columns_names = ['Queue size', 'Position', 'Event']
            return pd.DataFrame(np.zeros((self.nb_iter + 1, 3)),
                                             columns = columns_names)
        
    def update(self, h_0_mkt, h_0_stay, h_0, h_0_past, gamma= 0.2):
        # Initialize state variable
        self.initialize_variables()
        
        # Forward loop
        for i in range(self.nb_iter):
            # get next market event
            idx_event = self.getNextEvent()# *** 
            
            # get next state
            self.next_state, reward = self.getNext(idx_event)# ***

            # find optimal adjustment
            delta_mkt, delta_stay, delta = self.find_adjusts(h_0_stay, h_0_mkt, h_0, reward) # ***
            
            # update h_0s
            h_0[self.state.q, self.state.pos + 1] += gamma * delta
            h_0_stay[self.state.q, self.state.pos + 1] += gamma * delta_stay
            h_0_mkt[self.state.q, self.state.pos + 1] += gamma * delta_mkt

            h_0_past[self.state.q, self.state.pos + 1, 0] = delta_mkt
            h_0_past[self.state.q, self.state.pos + 1, 1] = delta_stay
            h_0_past[self.state.q, self.state.pos + 1, 2] = delta

            # print summary
            self.print_summary(i, idx_event)
            
            # update current state 
            self.state = self.process_state(self.next_state)
                   
        return h_0_mkt, h_0_stay, h_0, h_0_past
    
    def getNextEvent(self):
        idx_row = self.state.q
        intensities = self.intensity_values.loc[idx_row, self.columns_intensity]
        if (self.state.q == 1) and (self.state.pos == 1):# cancellation impossible when LAST one remaining in the queue
            intensities[1] = 0
        times = np.random.exponential(1/intensities)
        idx_event = times.argmin()
        
        return idx_event
    
    def getNext(self, idx_event):
        next_state = BookState(self.state.q, self.state.pos)
        
        # Buy limit order
        if idx_event == 0:
            next_state.q = min(self.state.q + 1, self.size_q - 1)
            reward = self.get_reward(next_state)
        # Buy cancel order
        elif idx_event == 1:
            if (self.state.q <= 1) and (self.state.pos <= 0) :
                next_state.q = max(self.state.q - 1, 0)
                reward = 0
            elif (self.state.q > 1) and (self.state.pos <= self.state.q - 1)  :
                next_state.q = self.state.q - 1
                reward = self.get_reward(next_state)
            elif (self.state.q > 1) and (self.state.pos == self.state.q) :
                next_state.q = self.state.q - 1
                next_state.pos = self.state.pos - 1
                reward = self.get_reward(next_state)
            else:
                raise ValueError("Cancellation impossible when q: {self.state.q},\
                                  and pos: {self.state.pos}")
        # Sell Market order
        elif idx_event == 2: 
            if (self.state.pos == 1):
                next_state.q = max(self.state.q - 1, 0)
                next_state.pos = 0
                reward = self.get_reward(next_state)
            elif (self.state.pos > 1):
                next_state.q = self.state.q - 1
                next_state.pos = self.state.pos - 1
                reward = self.get_reward(next_state)
            elif (self.state.pos <= 0):
                next_state.q = max(self.state.q - 1, 0)
                reward = 0
            else:
                raise ValueError("Market order impossible when q: {self.state.q},\
                                  and pos: {self.state.pos}")
                   
        return next_state, reward  
        
    def get_reward(self, state):# **
        # win when executed
        if state.pos ==  0: 
            return self.gain
        # cost of a market order
        elif state.pos ==  -1:
            return self.cost_out
        # cost of waiting
        else : 
            return self.cost_stay
        
    def find_adjusts(self, h_0_stay, h_0_mkt, h_0, reward):
        # adjustment market 
        delta_mkt = self.find_adjust_market(h_0_mkt)#**
        
        # adjustment stay 
        delta_stay = self.find_adjust_stay(h_0_stay, reward)# **
        
        # optimal adjustment
        delta = self.find_opti_adjust(h_0, reward)# **
        
        return delta_mkt, delta_stay, delta

    def find_adjust_stay(self, h_0_stay, reward):
        # the value of stay decision
        if (self.next_state.pos <= 0):
            h_stay = 0
        else:
            h_stay = h_0_stay[self.next_state.q, self.next_state.pos + 1]# TRANSLATE BECAUSE OF -1 is an execution by a market order
        
        delta_stay = reward + self.eta * h_stay - h_0_stay[self.state.q, self.state.pos + 1]
        return delta_stay

    def find_adjust_market(self, h_0_mkt):
        # reward market order
        if (self.next_state.pos >= 1): 
            q_after_mkt =  max(self.next_state.q - 1, 0)
            pos_after_mkt = -1
            state_after_market = BookState(q_after_mkt, pos_after_mkt)
            reward_mkt = self.get_reward(state_after_market)
        else:
            reward_mkt = 0
        
        delta_mkt = reward_mkt - h_0_mkt[self.state.q, self.state.pos + 1]
        return delta_mkt
    
    def find_opti_adjust(self, h_0, reward):
        # the value of stay decision
        if (self.next_state.pos <= 0):
            h_stay = 0
        else:
            h_stay = h_0[self.next_state.q, self.next_state.pos + 1]
        
        # reward market order
        if (self.next_state.pos >= 1): 
            q_after_mkt =  max(self.next_state.q - 1, 0)
            pos_after_mkt = - 1
            state_after_market = BookState(q_after_mkt, pos_after_mkt)
            reward_mkt = self.get_reward(state_after_market)
        else:
            reward_mkt = 0
        
        delta = max(reward_mkt, reward + self.eta * h_stay) \
                - h_0[self.state.q, self.state.pos + 1]
        return delta

    def process_state(self, next_state):
        if (next_state.pos <= 0):
            q = np.random.randint(1, self.size_q)
            pos = np.random.randint(1, q + 1)
            new_state = BookState(q, pos)
        else:
            new_state = next_state
        
        return new_state
        
    def getLoss(self, v, v_theo):
        return np.linalg.norm(np.nan_to_num(v) - np.nan_to_num(v_theo))

    def print_summary(self, i, idx_event):
        if self.write_history:
            self.book_history.loc[i + 1, 0] = self.next_state.q
            self.book_history.loc[i + 1, 1] = self.next_state.pos
            self.book_history.loc[i + 1, 2] = idx_event