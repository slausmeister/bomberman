import os
import pickle
import random

import numpy as np


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
    self.target_pos=None


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    """ # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) """

    self.logger.debug("Querying model for action.")

    # Look for a target if none is found
    
    coin_lock(self,game_state)
    return game_state['user_input']
    #return np.random.choice(ACTIONS, p=self.model)

def norm(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def coin_lock(self, game_state):
    if self.target_pos == None or self.target_pos not in game_state['coins']:
        coins = game_state['coins']

        coin_distance = [norm(coin,game_state['self'][3]) for coin in coins]

        if len(coin_distance)!=0:
            self.target_pos = coins[coin_distance.index(min(coin_distance))]
            print("Found a new target!")
        else:
            self.target_pos=None
            print("No target could be found")
        
        



def relative_position(entity, game_state, second=None):
    """
    Returns an array indicating the relative position of two objects
    
    """
    if second == None:
        return tuple(np.subtract(entity,game_state['self'][3]))
    else:
        return tuple(np.subtract(entity,second))

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

    # 1: Initialising coin potential
    coins = game_state['coins']
    coins = [norm(coin,game_state['self'][3]) for coin in coins]
    
    if len(coins)!=0:
        coins = [1/coin for coin in coins]

    #Sorting the array, and reshape it to lenght 9
    coins.sort()
    for n in range(len(coins), 9):
        coins.append(0)
    
    print(coins)
    
    stacked_channels = np.stack(coins)

# and return them as a vector
    return stacked_channels.reshape(-1)

    

    
