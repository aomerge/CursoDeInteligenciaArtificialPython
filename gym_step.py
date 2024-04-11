from argparse import Action
from email.errors import ObsoleteHeaderDefect
import imp, sys, os
import gym 
import ale_py
from gym.envs.classic_control import rendering


Qbert = gym.make('Qbert-v5')
Max_episode = 10
Mov_max = 500

for episode in range(Max_episode):
    obs = Qbert.reset()
    for step in range(Mov_max):
        Qbert.render()
        action = Qbert.step(Qbert.action_space.sample())
        next_state, reward, done, info = Qbert.step(action)
        obs = next_state

        if done is True :
             print('\n episode #{} terminado en {} steps'.format(episode, step+1)) 
             break
Qbert.close()    