import gym # type: ignore
from gym.wrappers import RecordVideo # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
import random 
import datetime
from argparse import ArgumentParser 

from libs.cnn import CNN
from libs.Perseptron import SimplePerceptron # type: ignore

from utils.decay_schedule import LinearDecaySchedule # type: ignore
from utils.experienceMemory import ExpirenceMemory, Experiences  # type: ignore
from utils.paramsManager import ParamsManager

## Parser Arguments
args = ArgumentParser("DeepQLearning")

args.add_argument("--params-file", type = str, help="Path to the file with the parameters", default = "./parameters.json")
args.add_argument("--env-name", type = str, help="Name of the environment", default = "CartPole-v1" )
args.add_argument("--id-gpu", type =int, help="Id of the GPU to use", default =0, metavar = "Id_GPU")
args.add_argument("--test", action = "store_true", help="Test the model", default=False )
args.add_argument("--reder", action = "store_true", help="Render the environment", default=False)
args.add_argument("--record", action = "store_true", help="Record the environment", default=False)
args.add_argument("--output-dir", type = str, help="Output directory for the video", default = "./trainer_Models/result/")

args = args.parse_args()

# Parameters 
manager = ParamsManager(args.params_file)
sumaryFilenamePrefix = manager.getAgentParams()['sumaryFilenamePrefix']
summaryFilename = sumaryFilenamePrefix + args.env_name + datetime.now().strftime("%y-%m-%d-%H%")
manager.exportAgentParams(summaryFilename + "/" +"agentParams.json")
manager.exportEnvParams (summaryFilename + "/" +"envParams.json")

## cont variables   
globalSteps = 0

# gpu or cpu
useCuda = manager.getAgentParams()['useCuda']
device = torch.device("cuda"+ str(args.id_gpu) if torch.cuda.is_available() and useCuda else "cpu")

# seed management
seed = manager.getAgentParams()['seed']
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available() and useCuda:
    torch.cuda.manual_seed_all(seed)    

class DeepQLearning(object):
    """
    Clase que implementa el algoritmo Q-Learning
    
    @param env: gym.Env: Entorno de OpenAI
    @param learning_rate: float: Tasa de aprendizaje
    @param gamma: float: Factor de descuento
    @param epsilon: float: Probabilidad de exploración
    @param epsilon_min: float: Probabilidad mínima de exploración
    @param epsilon_decay: float: Decaimiento de la probabilidad de exploración
    """
    def __init__(self, obsShape, actionShape ):        

        self.params = manager.getAgentParams()
        self.enviroment = manager.getEnvParams()
        self.gamma = self.params['gamma']
        self.learningRate = self.params['learningRate']
        self.bestReward = -float('inf')
        self.beastMean = -float('inf')
        self.trainingCompleted = 0
        self.MAX_NUM_EPISODES = self.params['maxNumEpisodes']
        self.STEPS_PER_EPISODE = self.params['stepsPerEpisode']
        
        if len(obsShape) == 1: 
            self.DQN = SimplePerceptron
        elif len(obsShape) == 3:
            self.DQN = CNN

        self.Q = self.DQN(obsShape, actionShape, device = device).to(device) # Red neuronal
        self.Q_optimize = torch.optim.Adam(self.Q.parameters(), lr= self.learningRate) # Optimizador        
        
        if self.params['useTargetNetwork']:
            self.Q_target = self.DQN(obsShape, actionShape, device = device).to(device) # Red neuronal            

        
        self.policy = self.epsilon_greedy_Q   
        self.epsilon_max = self.params['epsilonMax']
        self.epsilon_min = self.params['epsilonMin']
        self.epsilon_decay = LinearDecaySchedule(initialValue = self.epsilon_max, 
                                                    finalValue = self.epsilon_min, 
                                                    max_steps = self.params['epsilonDecay'])
        self.step_num = 0

        self.memory = ExpirenceMemory( capacity = int(self.params["experienceMemory"]) ) 
        
        """ self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") """

    def get_action(self, obs):
        """Obtención de la acción del agente"""
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs =  obs / 255.0
        if len(obs.shape) == 3: # es una imagen
            if obs.shape[2] < obs.shape[0]: # -> HWC -> CHW
                obs = obs.reshape(obs.shape[2], obs.shape[1], obs.shape[0])
            obs = np.expand_dims(obs, axis=0)
        
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):       
        """
        Esta función se encarga de seleccionar la acción del agente
        @param obs: np.array: Observación del entorno
        @return int: Acción seleccionada
        """ 
        self.step_num += 1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            action = np.random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device("cpu")).numpy())         

        return action
    
    def learn(self, obs, action, reward, next_obs, done):
        """
        Esta función se encarga de aprender del entorno
        @param obs: np.array: Observación del entorno
        @param action: int: Acción seleccionada
        @param reward: float: Recompensa obtenida
        @param next_obs: np.array: Observación del siguiente estado
        """
        if done: 
            td_target = reward + 0.0
        else:
            tensor_next_obs = torch.from_numpy(next_obs).float().to(self.device)        
            td_target = reward + self.gamma * torch.max(self.Q(tensor_next_obs)) # Cálculo del target temporal

        td_error = td_target - torch.nn.functional.mse_loss(self.Q(obs), td_target) # Cálculo del error temporal
        self.Q_optimize.zero_grad()
        td_error.backward()
        self.Q_optimize.step()

    def ReplaySave(self, batchSize = None):
        """
        Esta funcion se encarga de cargar y jugar con una muestra de experiencias
        @param batchSize: int: Tamaño de la muestra
        """
        batchSize = batchSize if batchSize is not None else self.params['replayBatchSize']
        experience = self.memory.sample(batchSize)
        self.learnFromExperience(experience)
        self.trainingCompleted += 1;

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
        nextObsBatch = torch.tensor(batchXP.new_state, dtype=torch.float32).to(self.device).float() / 255
        rewardBatch = torch.tensor(batchXP.reward, dtype=torch.float32).to(self.device)
        if self.enviroment["clipReward"]:
            rewardBatch = torch.sign(rewardBatch)
        doneBatch = torch.tensor(batchXP.done, dtype=torch.float32).to(self.device)
        not_doneBatch = 1 - doneBatch
        gamma_tensor = torch.full((len(nextObsBatch),), self.gamma, dtype=torch.float32, device=self.device)

        if self.params['useTargetNetwork']:
            if self.step_num % self.params['targetNetworkUpdateFreq'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())  
            td_target = rewardBatch + not_doneBatch * gamma_tensor * torch.max(self.Q_target(nextObsBatch), dim=1)[0]

        else : 
            # Calcular el valor Q para el siguiente estado
            next_state_values = self.Q(nextObsBatch).detach().max(1)[0]
            td_target = rewardBatch + not_doneBatch * gamma_tensor * next_state_values

        # Calcular el valor Q actual y el error de TD
        action_idx = self.Q(obsBatch).gather(1, actionBatch.unsqueeze(1))
        td_error = torch.nn.functional.mse_loss(action_idx, td_target.unsqueeze(1))

        # Backpropagation
        self.Q_optimize.zero_grad()
        td_error.backward()
        self.Q_optimize.step()

    def save(self, envName):
        """
        Esta función se encarga de guardar el modelo
        @param filename: str: Nombre del archivo
        """
        filename = self.params['saveDir']+"DQL_"+envName+".pth"
        agentState = {
            "Q": self.Q.state_dict(),
            "BestReward": self.bestReward,
            "BestRewardMean": self.beastMean
        }
        torch.save(agentState, filename)
        print("Modelo guardado en {}".format(filename))

    def load(self, envName):
        """
        Esta función se encarga de cargar el modelo
        @param envName: str: Nombre del archivo
        """
        filename = self.params['loaDir']+"DQL_"+envName+".pth"
        agentState = torch.load(filename, map_location = lambda storage, loc: storage)
        self.Q.load_state_dict(agentState["Q"])
        self.Q.to(device)
        self.bestReward = agentState["BestReward"]
        self.beastMean = agentState["BestRewardMean"]
        print("Modelo cargado desde {}".format(filename))
        
                     

if __name__ == "__main__":
    envConfig = manager.getEnvParams()
    envConfig["env_name"] = args.env_name

    if args.test : 
        envConfig["episodicLife"] = False

    rewardType = "LIFE" if envConfig["episodicLife"] else "GAME"
    env = gym.make()
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