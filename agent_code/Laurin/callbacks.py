import os
import pickle
import random

import numpy as np

from agent_code.Laurin.func import *

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.
    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    #setting up coordinates for rotating the board
    setup_coords(self)

    if not self.train:
        try:
            self.coin_states_actions = pickle.load(open("coin_states_actions.pt", "rb"))
        except:
            raise Exception("coin_states_actions could not be loaded")
            self.coin_states_actions = None
            self.logger.debug(f"coin_states_actions could not be loaded")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    new_game_state, state= game_state_transformer(self, game_state)

    
    ownpos = new_game_state["self"][3]
    coin, min = get_nearest_coin_position(ownpos, new_game_state["coins"])
    

    if coin != None:
        state_id = str(ownpos[0]) + str(ownpos[1]) + str(coin[0]) + str(coin[1])
        actions = ["WAIT"]
        val = -1000

        for action in ["LEFT", "RIGHT", "UP", "DOWN"]:
            if self.coin_states_actions[state_id + action] == val:
                actions.append(action)

            elif self.coin_states_actions[state_id + action] > val:
                val = self.coin_states_actions[state_id + action]
                actions = [action]
        
        if actions == ["WAIT"]:
            return np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"])
        elif len(action) > 1:
            a = np.random.choice(actions)
        else:
            a = actions[0]
    
    else:
        self.logger.info("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"])


    ## EXPLORATION
    random_prob = 0.3
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"])

    self.logger.debug(f"{a}")
    if a in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    if state == "ru":
                        a = self.order_ru[a]
                    if state == "rd":
                        a = self.order_rd[a]
                    if state == "ld":
                        a = self.order_ld[a]
    self.logger.debug(f"{a}")
    return a


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*
    Converts the game state to the input of your model, i.e.
    a feature vector.
    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
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