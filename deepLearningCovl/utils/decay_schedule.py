class LinearDecaySchedule(object):
    def __init__(self, initialValue: float, finalValue: float, max_steps: int):
        assert initialValue > finalValue, "initialValue should be greater than finalValue."
        self.initialValue = initialValue
        self.finalValue = finalValue
        self.decay_factor = (initialValue - finalValue) / max_steps

    def __call__(self, step_num: int):
        current_value = self.initialValue - step_num * self.decay_factor 
        if current_value < self.finalValue:
            current_value = self.finalValue
        return current_value
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt # type: ignore
    estepsEpisode = 300
    maxNumEpisode =  1000
    schedule = LinearDecaySchedule(initialValue = 1.0, finalValue = 0.01, max_steps = 0.8 * maxNumEpisode * estepsEpisode)
    epsilons = [schedule(step) for step in range(maxNumEpisode * estepsEpisode)]
    plt.plot(epsilons)
    plt.show()