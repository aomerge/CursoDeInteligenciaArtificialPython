import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame

# Definimos la red neuronal que aproxima la función Q
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # CartPole tiene un espacio de estado de 4 dimensiones
        self.fc2 = nn.Linear(64, 2)  # Dos acciones posibles: izquierda o derecha

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Inicializamos el entorno, la red y el optimizador
env = gym.make('CartPole-v1', render_mode='human')
q_network = QNetwork()
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# Entrenamiento
num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        try:
            # Extraemos solo el array NumPy del estado, asumiendo que el estado es una tupla
            # donde el primer elemento es el array de NumPy que representa el estado
            state_array = state[0] if isinstance(state, tuple) else state
            print(f"Estado actual: {state_array}")
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            ...
        except ValueError as e:
            print(f"Error al convertir el estado a tensor: {e}")
            print(f"Estado problemático: {state}")
            break


# Cerrar el entorno
env.close()
