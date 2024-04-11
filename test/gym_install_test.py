import gym
def test_gym_install():
    try:
        env = gym.make('MountainCar-v0', render_mode='human')
        env.reset()
        for _ in range(1000):  
            env.render()                  
            env.step(env.action_space.sample())
        env.close()
    except ZeroDivisionError as error:    
        print("Se produjo un error:", error)
    
if __name__ == "__main__":
    test_gym_install()