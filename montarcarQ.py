import gym
import numpy as np
import random
import pygame

# Configuraciones iniciales
MAX_NUM_EPISODES = 100
STEPS_PER_EPISODE = 100
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

class QLearning(object):
    def __init__(self, car):
        self.obs_shape = car.observation_space.shape
        self.obs_high = car.observation_space.high
        self.obs_Low = car.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width = (self.obs_high - self.obs_Low) / self.obs_bins
        self.action_shape = car.action_space.n
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))  # Matriz de 31 x 31 x 3
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):        
        normalized = (obs - self.obs_Low) / self.obs_width
        discretized = np.clip(normalized.astype(int), 0, self.obs_bins)
        return tuple(discretized)

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Decremento de epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        # Selección de la acción basada en la política epsilon-greedy
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target - self.Q[discrete_obs][action]
        self.Q[discrete_obs][action] += self.alpha * td_error

def train(agent, car):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = car.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            
            # Ajustar aquí para manejar el extra valor devuelto
            result = car.step(action)
            next_obs, reward, done, _, info = result  # Asume que el cuarto valor no se necesita

            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward
        print(f'Episodio número {episode} con recompensa {total_reward}, mejor recompensa {best_reward}, epsilon {agent.epsilon}')

    return np.argmax(agent.Q, axis=2)


if __name__ == "__main__":
    car = gym.make("MountainCar-v0", render_mode='human')
    agent = QLearning(car)
    learned_policy = train(agent, car)    
    car.close()

