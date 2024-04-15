"""
Created on 2024-01-20
@author: aomerge
"""
# Import the necessary libraries
import gym
import numpy as np
import torch
import torch.nn as nn
import gym.spaces
import sys

def print_spaces(space):
    print(space)
    if isinstance(space, gym.spaces.Box):
        print(f"Space low: {space.low}")
        print(f"Space high: {space.high}")

if __name__ == "__main__":
    # Create the environment
    env = gym.make(sys.argv[1] if len(sys.argv) > 1 else "CartPole-v1")

    # Print the observation space
    print("Observation space:")
    print_spaces(env.observation_space)

    # Print the action space
    print("Action space:")
    print_spaces(env.action_space)

    try:
        # Print the observation space sample
        print("Space Action Sample:")
        print(env.unwrapped.get_action_meanings())
    except:
        print("Cannot sample from observation space")
    # Close the environment
    env.close()

