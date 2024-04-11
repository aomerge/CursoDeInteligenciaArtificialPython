import gym
from gym.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt

#formula de faiman
#Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))

# Configuraciones iniciales
MAX_NUM_EPISODES = 1000 # Número máximo de episodios
STEPS_PER_EPISODE = 100 # Número de pasos por episodio
EPSILON_MIN = 0.05 # Epsilon mínimo
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE # Número máximo de pasos
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps # Decremento de epsilon
ALPHA = 0.5 # Tasa de aprendizaje
GAMMA = 0.80 # Factor de descuento
NUM_DISCRETE_BINS = 30 # Número de bins para discretizar el espacio de observación


# Clase QLearning
class QLearning(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_Low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width = (self.obs_high - self.obs_Low) / self.obs_bins
        self.action_shape = env.action_space.n
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))  # Matriz de 31 x 31 x 3
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):
        obs_array = obs[0] if isinstance(obs, tuple) and isinstance(obs[0], np.ndarray) else obs

        normalized = (obs_array - self.obs_Low) / self.obs_width  # Normalización
        discretized = np.clip(normalized.astype(int), 0, self.obs_bins - 1)  # Discretización
        return tuple(discretized)

    def get_action(self, obs):
        """Obtención de la acción del agente"""
        discrete_obs = self.discretize(obs)
        # Decremento de epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        # Selección de la acción basada en la política epsilon-greedy
        if np.random.random() > self.epsilon: # Explotación
            return np.argmax(self.Q[discrete_obs]) 
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs) # Discretización de la observación actual
        discrete_next_obs = self.discretize(next_obs) # Discretización de la siguiente observación
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs]) # Cálculo del target temporal
        td_error = td_target - self.Q[discrete_obs][action] # Cálculo del error temporal
        self.Q[discrete_obs][action] += self.alpha * td_error # Actualización de la Q-table

    def train(agent, env):
        best_reward = -float('inf')
        for episode in range(MAX_NUM_EPISODES):
            done = False
            obs = env.reset()
            total_reward = 0.0            
            while not done:
                action = agent.get_action(obs) # selection action
                next_obs, reward, done, _, info = env.step(action)
                agent.learn(obs, action, reward, next_obs)
                obs = next_obs
                total_reward += reward
            if total_reward > best_reward:
                best_reward = total_reward
        
            print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode, total_reward, best_reward, agent.epsilon))
        return np.argmax(agent.Q, axis=2)

    def save(self, filename):
        np.save(filename, self.Q)

    def load(self, filename):
        self.Q = np.load(filename)
        print("Q-table loaded from", filename)

def test(agent, env):
    obs = env.reset()
    total_reward = 0.0
    while True:
        action = agent.get_action(obs)
        next_obs, reward, done, _,info = env.step(action)
        total_reward += reward
        if done:
            break
        obs = next_obs
    print("Total reward in test:", total_reward)
    env.close()

if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    agent = QLearning(env)  # Ensure QLearning is properly defined
    learned_policy = agent.train(env)  # Assuming QLearning has a train method
    agent.save("Q_table.npy")
    monitorPath = "./monitor_output"
    env = RecordVideo(env, monitorPath)
    test(agent, env)    
    env.close()
