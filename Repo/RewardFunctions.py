class ParameterizedDiscreteReward:
    def __init__(self, params=None):
        """
        params = parameter dictionary
        """
        if params is None:
            self.parameters = {'outofbounds': 0,
                               'finalgoal': 0,
                               'timepenalty': 0,
                               'withinbounds': 0,
                               'matchenv': 0,
                               'bumpless': 0}
        else:
            self.parameters = params

    def set_coefficients(self, params):
        """
        params = parameter dictionary
        """
        self.parameters = params

    def calculate_reward(self, reward_tags):

        reward = 0

        for key in reward_tags:
            reward = reward + reward_tags[key] * self.parameters[key]

        return reward
