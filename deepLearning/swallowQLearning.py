import gym # type: ignore
from gym.wrappers import RecordVideo # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from libs.Perseptron import SimplePerceptron # type: ignore
from utils.decay_schedule import LinearDecaySchedule
import random 
from utils.experienceMemory import ExpirenceMemory, Experiences

# Configuraciones iniciales
MAX_NUM_EPISODES = 10000 # Número máximo de episodios
STEPS_PER_EPISODE = 100 # Número de pasos por episodio

class SwallowQLearning(object):
    """
    Clase que implementa el algoritmo Q-Learning
    
    @param env: gym.Env: Entorno de OpenAI
    @param learning_rate: float: Tasa de aprendizaje
    @param gamma: float: Factor de descuento
    @param epsilon: float: Probabilidad de exploración
    @param epsilon_min: float: Probabilidad mínima de exploración
    @param epsilon_decay: float: Decaimiento de la probabilidad de exploración
    """
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
        self.memory = ExpirenceMemory( capacity = int(1e6) ) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, obs):
        """Obtención de la acción del agente"""
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):       
        """
        Esta función se encarga de seleccionar la acción del agente
        @param obs: np.array: Observación del entorno
        @return int: Acción seleccionada
        """ 
        if random.random() < self.epsilon_decay(self.step_num):
            action = np.random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device("cpu")).numpy())         
        return action

    def learn(self, obs, action, reward, next_obs):
        """
        Esta función se encarga de aprender del entorno
        @param obs: np.array: Observación del entorno
        @param action: int: Acción seleccionada
        @param reward: float: Recompensa obtenida
        @param next_obs: np.array: Observación del siguiente estado
        """
        tensor_next_obs = torch.from_numpy(next_obs).float().to(self.device)        
        td_target = reward + self.gamma * torch.max(self.Q(tensor_next_obs)) # Cálculo del target temporal
        td_error = td_target - torch.nn.functional.mse_loss(self.Q(obs), td_target) # Cálculo del error temporal
        self.Q_optimize.zero_grad()
        td_error.backward()
        self.Q_optimize.step()

    def ReplaySave(self, batchSize):
        """
        Esta funcion se encarga de cargar y jugar con una muestra de experiencias
        @param batchSize: int: Tamaño de la muestra
        """
        experience = self.memory.sample(batchSize)
        self.learnFromExperience(experience)

    def validate_and_transform_observations(self, observations):
        # Asegurarse de que todos los elementos son arrays de NumPy y tienen la misma forma
        processed_obs = []
        first_elem_shape = None
        
        for elem in observations:
            if isinstance(elem, tuple):
                elem = elem[0]  # Asumiendo que el array de NumPy está en la primera posición de la tupla
            if not isinstance(elem, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got {type(elem).__name__}")
            
            if first_elem_shape is None:
                first_elem_shape = elem.shape
            elif elem.shape != first_elem_shape:
                raise ValueError("All observation arrays must have the same shape.")
            
            processed_obs.append(elem)
        
        # Convertir la lista de arrays de NumPy a un único array de NumPy
        obs_array = np.stack(processed_obs)
        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        return obs_tensor
    
    def learnFromExperience(self, experiences):
        """
        Esta función se encarga de aprender de una experiencia
        @param experiences: list: Experiencia a aprender
        """
        batchXP = Experiences(*zip(*experiences))        
        # Convertir las observaciones a tensores y enviarlas al dispositivo

        obsBatch = self.validate_and_transform_observations(batchXP.obs)
        #obsBatch = torch.tensor(batchXP.obs[0], dtype=torch.float32).to(self.device)
        actionBatch = torch.tensor(batchXP.action, dtype=torch.long).to(self.device)  # Asegúrate de que es long para indexing
        nextObsBatch = torch.tensor(batchXP.new_state, dtype=torch.float32).to(self.device)
        rewardBatch = torch.tensor(batchXP.reward, dtype=torch.float32).to(self.device)
        doneBatch = torch.tensor(batchXP.done, dtype=torch.float32).to(self.device)
        not_doneBatch = 1 - doneBatch
        gamma_tensor = torch.full((len(nextObsBatch),), self.gamma, dtype=torch.float32, device=self.device)


        # Calcular el valor Q para el siguiente estado
        next_state_values = self.Q(nextObsBatch).detach().max(1)[0]
        td_target = rewardBatch + not_doneBatch * gamma_tensor * next_state_values

        # Calcular el valor Q actual y el error de TD
        current_q_values = self.Q(obsBatch).gather(1, actionBatch.unsqueeze(1))
        td_error = torch.nn.functional.mse_loss(current_q_values, td_target.unsqueeze(1))

        # Backpropagation
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
            agent.memory.Store(Experiences(obs, action, reward, next_obs, done))
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
                if agent.memory.getSize() > 100:
                    agent.ReplaySave(32)
                break
        
    env.close()