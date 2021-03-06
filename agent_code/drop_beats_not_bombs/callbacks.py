from agent_code.drop_beats_not_bombs.func import *
import os
import pickle
from collections import deque
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((6,7),dtype=float)
        self.history = deque([], 20)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.history = deque([], 20)


def act(self, game_state: dict) -> str:
    self.history.append(game_state['self'][3])
    acts=np.array(ACTIONS)
    eps = 0.1
    X = state_to_features(game_state)
    moves = possible_actions(self,game_state)
    beta = self.model
    q_values = []
    
    for mov in moves:
        index = np.where(acts==mov)[0][0]         
        q_hat = X@beta[index]
        q_values.append(q_hat)

    q_values = np.array(q_values)
    highest_value = np.amax(q_values)
    value_indices = np.where(q_values==highest_value)[0]
    
    j=np.random.choice(value_indices,1)[0]

    action_greedy = moves[j]

    if self.train:
        actual_action = np.random.choice([action_greedy,np.random.choice(moves,1)],1,p=[1-eps,eps])
        return actual_action
    else:
        return action_greedy
    



