import gym

enviroment = gym.make("ALE/Qbert-v5", render_mode="human")
enviroment.metadata['render.modes'] = "human"
enviroment.metadata["render_fps"] = 30
num_episodes = 20
steps_per_episode = 200

for i_episode in range(num_episodes):
    observation = enviroment.reset()
    for t in range(steps_per_episode):        
        action = enviroment.action_space.sample()
        nextState, reward, done, extra_info, info = enviroment.step(action)
        obs= nextState
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
enviroment.close()
