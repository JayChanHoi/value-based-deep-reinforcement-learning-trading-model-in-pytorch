class Environment():
    def __init__(self, data, history_length=90):
        self.data = data
        self.history_length = history_length
        self.reset()

    def reset(self):
        self.t = 0
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_length)]
        # return the state/observation representation
        return [self.position_value] + self.history

    def __call__(self, action):
        reward = 0
        profits = 0
        hold_size = 0
        hold_amount = 0

        if action == 1:
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif action == 2:
            if len(self.positions) == 0:
                reward = -1
            else:
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['Close'] - p) -1
                # actually, this provides a way to make the reward be stochastic
                reward_signal = profits / sum(self.positions)

                # in this case reward is deterministic
                if reward_signal < 0:
                    reward = -1
                else:
                    reward = 1
                hold_size = len(self.positions)
                hold_amount = sum(self.positions)
                self.profits += profits
                self.positions = []

        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p) - 1
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'])

        return [self.position_value] + self.history, reward, profits, hold_size, hold_amount