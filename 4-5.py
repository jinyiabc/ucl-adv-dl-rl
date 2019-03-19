#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:43:37 2019

@author: jinyi
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize the state value
value = np.zeros(101)
value[100] = 1
state = np.arange(0,101)
policy = np.arange(0,101)
theta = 1e-10
state_ = state[1:100]
p_head = 0.40
p_tail = 1- p_head
# initialize state value
def function_4_5():
    global state_
    global value
    global policy    
    
    while True:
        dif = 0
        for i in state_:
            v = np.copy(value)
            value, max_action = expected_value(i, value)
            policy[i] = max_action
            dif += max(0, abs(v-value).sum())
        print("dif:",dif)
        if dif < theta:
#            print(policy)
#            print(value)
            
            plt.figure(figsize=(10, 20))
    
            plt.subplot(2, 1, 1)
            plt.plot(value)
            plt.xlabel('Capital')
            plt.ylabel('Value estimates')
        
            plt.subplot(2, 1, 2)
            plt.scatter(state, policy)
            plt.xlabel('Capital')
            plt.ylabel('Final policy (stake)')
        
            plt.savefig('../images/figure_4_3.png')
            plt.close()
            break
    
def expected_value(state, value):
    opt_action = temp = 0
    new_value = np.copy(value)
    actions = np.arange(0, min(state, 100-state)+1)
    action_return = []

    for action in actions:
        lose = state-action
        win = state+action
        temp = p_head*(value[win]) + p_tail*value[lose]
#        if temp > value[state]:
#            new_value[state] = temp
#            opt_action = action
        action_return.append(temp)
    idx = np.argmax(action_return)
    new_value[state] = action_return[idx]
    opt_action = actions[idx]
    return new_value, opt_action
            
if __name__ == '__main__':
    function_4_5()
      
        