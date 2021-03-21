import os
import pickle
import random

import numpy as np
from agent_code.slausmeister.func import *


#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

def setup(self):
    self.round = 1
    if not self.train:
        try:
            self.state_action = pickle.load(open("state_action.pt", "rb"))
        except:
            raise Exception("coin_states_actions could not be loaded")
            self.state_action = None
            self.logger.debug(f"coin_states_actions could not be loaded")
    self.order_rev_ru = {"LEFT": "UP", "RIGHT": "DOWN", "UP": "RIGHT", "DOWN": "LEFT"}
    self.order_rev_ld = {"LEFT": "DOWN", "UP": "LEFT", "RIGHT": "UP", "DOWN": "RIGHT"}
    self.order_rev_rd = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}



def act(self, game_state: dict) -> str:
    state, quad = oriented_state(self,game_state)
    current_id = state_identification(state)

    # Selection an action
    val_actions = [(self.state_action[current_id + action], action) for action in ACTIONS]

    max(val_actions,key=lambda item:item[0])
    if len(val_actions)>1:
        suggestion = random.choice(val_actions)[1]


    # Reversing state rotation
    if quad == "ru":
        suggestion = self.order_rev_ru[suggestion]
    if quad == "rd":
        suggestion = self.order_rev_rd[suggestion]
    if quad == "ld":
        suggestion = self.order_rev_ld[suggestion]

    self.round = self.round + 1
    return suggestion
 

def state_to_features(game_state: dict) -> np.array:
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
