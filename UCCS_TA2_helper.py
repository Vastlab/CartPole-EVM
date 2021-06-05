#UCCS TA 2 helper 
import pdb
import numpy as np
import gym
from gym import make

import os
import random
import time
from utils import rollout
import time
import pickle
#import cv2
import PIL
import torch
import json
import argparse
from collections import OrderedDict
from functools import partial
from torch import Tensor
import torch.multiprocessing as mp
from my_lib import *
from vast.opensetAlgos.EVM import EVM_Training , EVM_Inference, EVM_Inference_simple_cpu
from vast import activations
from statistics import mean
import gc
import random
import csv


try:
  torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
  pass


try:
  mp.set_start_method('spawn', force=True)
except RuntimeError:
  pass



class  UCCSTA2():

    def __init__(self):

        #calibrated values for KL for cartpole wth one-step lookahead
        self.KL_threshold = 1        
        self.KL_val = 0
        self.num_epochs=200
        self.num_dims=4
        if(self.num_dims==4):
          self.mean_train=  0
          self.stdev_train =  0.0
          self.prob_scale=2    # need to scale since EVM probability on perfect training data is .5  because of small range and tailsize
        else:
          self.mean_train=  .198
          self.stdev_train =  0.051058052318592555
          self.prob_scale=1   #probably do need to scale but not tested sufficiently to see what it needs.           
          
        self.cnt = 0
        self.worldchanged=0        
        #from WSU "train".. might need ot make this computed.
#        self.mean_train=  0.10057711735799268
#       self.stdev_train = 0.00016        
        self.problist = []
        self.maxprob = 0     
        self.expected_backone = np.zeros(4)
#        self.expected_backtwo = np.zeros(4)                
        self.episode=0
        self.debugstring=""
        # Create prediction environment
        gym.envs.registration.register(id='CartPoleSwingUp-v0',
                                       entry_point='carty_cartpole_swingup:CartPoleSwingUp')
        self.env_prediction = gym.make('CartPoleSwingUp-v0')
        
        with open('evm_config.json', 'r') as json_file:
            evm_config = json.loads(json_file.read())
            cover_threshold = evm_config['cover_threshold']
            distance_multiplier = evm_config['distance_multiplier']
            tail_size = evm_config['tail_size']
            distance_metric =  evm_config['distance_metric']
            torch.backends.cudnn.benchmark=True
            args_evm  = argparse.Namespace()
            args_evm.cover_threshold = [cover_threshold]
            args_evm.distance_multiplier = [distance_multiplier]
            args_evm.tailsize = [tail_size]
            args_evm.distance_metric = distance_metric
            args_evm.chunk_size = 200
            filename = "evm_cosine_lookahead4dim1step_tail_200_ct_0.1_dm_0.5.pkl"
            if(self.num_dims==4):
              filename = "evm_cosine_lookahead4dim_tail_40000_ct_0.8_dm_0.55.pkl"
            else:
#              filename="evm_cosine_lookahead8dim1step_expected_dif_tail_1000_ct_0.1_dm_0.5.pkl"
              filename="evm_cosine_lookahead8dim_tail_40000_ct_0.8_dm_0.55.pkl"
#filename = "evm_cosine_lookahead8dim1step_tail_200_ct_0.1_dm_0.5.pkl"                        
            
#            filename = f"evm_{distance_metric}_lookahead4dim_tail_{tail_size}_ct_{cover_threshold}_dm_{distance_multiplier}.pkl"
#            filename = f"evm_{distance_metric}_lookahead4dim_tail_{tail_size}_ct_{cover_threshold}_dm_{distance_multiplier}.pkl"
            evm_model = pickle.load( open( filename, "rb" ) )
            self.evm_inference_obj = EVM_Inference_simple_cpu(args_evm.distance_metric, evm_model)
        return
    

    def reset(self,episode):
        self.problist = []
        self.maxprob = 0     
        self.cnt = 0        
        self.episode=episode
        self.worldchanged=0                
        
    # Take one step look ahead, return predicted environment and step
    # env should be a carty_cartpole_swingup environment
    def takeOneStep(self,state_given, env, pertub=False):
        observation = env.set_state(state_given)
        action, _ = env.get_best_action(observation)
        #if doing well pertub it so we can better chance of detecting novelties
        ra = int(state_given[0]*10000) %4
        if(pertub and
           (ra ==0 ) and
            ((abs(state_given[0]) < .2)
              and  (abs(state_given[1]) < .25)
              and (abs(state_given[2]) < .05)
              and (abs(state_given[3]) < .1))):
            if(action ==1): action = 0
            else: action = 1
            #print("Flipped Action, state=",state_given)

        expected_state, _, _, _ = env.step(action)
        return action, expected_state


    def kullback_leibler(self,mu, sigma, m, s):
        '''
        Compute Kullback Leibler with Gaussian assumption of training data
        mu: mean of test batch
        sigm: standard deviation of test batch
        m: mean of all data in training data set
        s: standard deviation of all data in training data set
        return: KL ditance, non negative double precison float
        '''
        sigma = max(sigma,.0000001)
        s = max(s,.0000001)        
        kl = np.log(s/sigma) + ( ( (sigma**2) + ( (mu-m) **2) ) / ( 2 * (s**2) ) ) - 0.5
        return kl


    def world_change_prob(self):
        mu = np.mean(self.problist)
        sigma = np.std(self.problist)
        if(len(self.problist) < 3): return 0;
        if(sigma == 0):
          if(mu==self.mean_train): return 0;
          else:
            self.worldchanged = 1;            
            return           self.worldchanged  

#        pdb.set_trace()
        self.KL_val =  self.kullback_leibler( mu, sigma,self.mean_train, self.stdev_train)
        KLscale = (self.num_epochs+1-self.episode/2)/self.num_epochs  #decrease scale (increase sensitvity)  from start down to  1/2
        prob = min(1.0, KLscale*self.KL_val/(2 *self.KL_threshold))
#        self.worldchanged = max(prob,self.worldchanged)
        self.worldchanged = prob
        return self.worldchanged
    
        
    
    def process_instance(self,actual_state):
        pertub = (self.cnt > 5) and (self.maxprob < .5)
        action, expected_state = self.takeOneStep(actual_state, self.env_prediction,pertub)        

        data_val = self.expected_backone
        self.expected_backone = expected_state
        self.cnt += 1        
        if(self.cnt <3):    #skip prob estiamtes for the first ones as we need history to get prediction
          self.debugstring='Early Instance: actual_state={}, next={}, dataval={}, '.format(actual_state, expected_state, data_val)         
          return action
        
        difference_from_expected = data_val - actual_state  #next 4 are the difference between expected and actual state after one step, i.e.
        current = difference_from_expected        
#        for i in difference_from_expected:
#            current.append(i)
        data_tensor = torch.from_numpy(np.asarray(current))
        probs = self.evm_inference_obj(data_tensor)
        probability = self.prob_scale*(probs.numpy()[0])-1  #probably of novelty so knowns have prob 0,  unknown prob 1.
        self.maxprob = max(probability,self.maxprob)
        self.debugstring ='Instance: cnt={},actual_state={}, next={},  current/diff={},NovelProb={}'.format(self.cnt,actual_state, expected_state, current,probability)         
#        if(probability >0): pdb.set_trace()
        self.problist.append(probability)
        

        return action
        

