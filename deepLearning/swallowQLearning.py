import gym # type: ignore
from gym.wrappers import RecordVideo # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from libs.Perseptron import SimplePerceptron # type: ignore
from utils.decay_schedule import LinearDecaySchedule
import random 

# Configuraciones iniciales
MAX_NUM_EPISODES = 10000 # Número máximo de episodios
STEPS_PER_EPISODE = 100 # Número de pasos por episodio

class SwallowQLearning(object):
    def __init__(self, env, learning_rate = 0.01, gamma = 0.99, epsilon = 1.0, epsilon_min = 0.05, epsilon_decay = 0.5):
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.n
        self.Q = SimplePerceptron(self.obs_shape, self.action_shape) 
        self.Q_optimize = torch.optim.Adam(self.Q.parameters(), lr= learning_rate) # Optimizador        
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initialValue = self.epsilon_max, 
                                                    finalValue = self.epsilon_min, 
                                                    max_steps = epsilon_decay * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q    

    def get_action(self, obs):
        """Obtención de la acción del agente"""
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):        
        if random.random() < self.epsilon_decay(self.step_num):
            action = np.random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device("cpu")).numpy())         
        return action

    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs)) # Cálculo del target temporal
        td_error = td_target - torch.nn.functional.mse_loss(self.Q(obs), td_target) # Cálculo del error temporal
        self.Q_optimize.zero_grad()
        td_error.backward()
        self.Q_optimize.step()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = SwallowQLearning(env)
    firstEpisode = True
    episodeRewards = list()
    for ep in range(MAX_NUM_EPISODES):
        obs = env.reset()
        total_reward = 0.0
        for t in range(STEPS_PER_EPISODE):
            action = agent.get_action(obs)
            next_obs, reward, done, _,info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
            if done is True:
                if firstEpisode:
                    maxReward = total_reward
                    firstEpisode = False
                episodeRewards.append(total_reward)
                if total_reward > maxReward:
                    maxReward = total_reward
                print ("Episodio#:{} finalizado con {} iteraciones. Recompensa:{} , recompenza media:{},  Mejor recompensa:{}" .format(ep, t+1, total_reward, np.mean(episodeRewards), maxReward))
                break
        
    env.close()