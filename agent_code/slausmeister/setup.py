def setuo_state(self):
    self.state_action = {}
    for x in range(1, 9):
        for y in range(1, 9):
            for m in range(1, 17):
                for n in range(1, 17):
                    self.state_action[str(x) + str(y) + str(m) + str(n) + "LEFT"] = 1
                    self.state_action[str(x) + str(y) + str(m) + str(n) + "RIGHT"] = 1
                    self.state_action[str(x) + str(y) + str(m) + str(n) + "UP"] = 1
                    self.state_action[str(x) + str(y) + str(m) + str(n) + "DOWN"] = 1

def setup_relative_state(self):
    self.state_action = {}
    for m in range(1, 31):
        for n in range(1, 31):
            self.state_action[str(m) + str(n) + "LEFT"] = 1
            self.state_action[str(m) + str(n) + "RIGHT"] = 1
            self.state_action[str(m) + str(n) + "UP"] = 1
            self.state_action[str(m) + str(n) + "DOWN"] = 1