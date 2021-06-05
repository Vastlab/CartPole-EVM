#TA 2 BASE
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



import UCCS_TA2_helper
from  UCCS_TA2_helper import UCCSTA2
UCCS = UCCSTA2()


try:
  torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
  pass


try:
  mp.set_start_method('spawn', force=True)
except RuntimeError:
  pass

number_of_classes = 2

n_cpu = int(os.cpu_count()*0.8)


SEED = 1
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
#tf.random.set_seed(SEED)



env_to_use = 'CartPole-v0'
print('ENV TO USE', env_to_use)
env = gym.make(env_to_use)

noveltyStdMn = []
state_and_dec = []
X = []; Y = [];
x = env.reset()
numSteps = 0
KLlist = []
currentRunProb = []

nruns = 200
for k in range(nruns):
    actual_state = env.reset()
    numSteps = 0
    state_and_dec = []
    currentRunProb = []
    for i in range(200):
        # Predict steps
      action = UCCS.process_instance(actual_state)            
      if(UCCS.cnt < 25): print(UCCS.debugstring)        
      actual_state, r, done, _ = env.step(action)          # Take the predicted best action to get next actual state
      if done:
        if(UCCS.cnt < 199): print("!!!!!!!!!!!!!!!!!!!!!Steps only:", numSteps)

        print (UCCS.problist)
        mu = np.mean(UCCS.problist[3:])
        sigma = np.std(UCCS.problist[3:])
        print(mu,sigma)
        kl = UCCS.kullback_leibler( mu, sigma,UCCS.mean_train, UCCS.stdev_train)
        KLlist.append(float(kl))
        UCCS.episode += 1
        print("Steps, KL/WC",UCCS.cnt,kl, UCCS.world_change_prob())
        UCCS.cnt=0
        UCCS.problist=[]
        UCCS.reset(0)

        break

            

#    KLlist.append(currentRunProb)

#pdb.set_trace()

print("mean/stdev KL", np.mean(KLlist),np.std(KLlist))

