import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from agent_code.slausmeister.func import *
from agent_code.slausmeister.setup import *


def setup_training(self):
    self.order_ld = {"LEFT": "UP", "RIGHT": "DOWN", "UP": "RIGHT", "DOWN": "LEFT"}
    self.order_rd = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
    self.order_ru = {"LEFT": "DOWN", "UP": "LEFT", "RIGHT": "UP", "DOWN": "RIGHT"}

    try:
        self.state_action = pickle.load(open("state_action.pt", "rb"))
    except:
        self.state_action = None
        self.logger.debug(f"state_action could not be loaded in train.py")


    if self.state_action == None:
        #setup_state(self)
        setup_relative_state(self)
                        
    


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    try:
        #old, old_quad = oriented_state(self,old_game_state)
        #new, new_quad = oriented_state(self,new_game_state)
        reward = 0

        old, old_quad = oriented_relative_state(self,old_game_state)
        new, new_quad = oriented_relative_state(self,new_game_state)


        if self_action in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    if old_quad == "ru":
                        self_action = self.order_ru[self_action]
                    if old_quad == "rd":
                        self_action = self.order_rd[self_action]
                    if old_quad == "ld":
                        self_action = self.order_ld[self_action]

        reward = reward + reward_from_events(self, events)

        #if taxi(np.nonzero(old[0]),np.nonzero(old[1]))[0] > taxi(np.nonzero(new[0]),np.nonzero(new[1]))[0]:
        #    reward = reward +30

        if taxi(np.nonzero(old),(15,15))[0] > taxi(np.nonzero(new),(15,15))[0]:
            reward = reward +30

        self.state_action[relative_state_identification(old) + self_action] = reward + self.state_action[relative_state_identification(old) + self_action]
    except:
        print(f"Number of itterations: {self.round}")

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("state_action.pt", "wb") as file:
        pickle.dump(self.state_action, file)
    self.round = 0

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -5

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
