from agent_code.nlb_agent.func import nearest_coin
import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONBEGIN = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
CLOSER = 'CLOSER'
FURTHER = 'FURTHER'


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

alpha=0.1
gamma=0.9

def setup_training(self):
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if old_game_state == None:
        pass
    else:
        if nearest_coin(old_game_state)[0] > nearest_coin(new_game_state)[0]:
            events.append(CLOSER)
        if nearest_coin(old_game_state)[0] < nearest_coin(new_game_state)[0]:
            events.append(FURTHER)

        R = reward_from_events(self,events)

        beta = self.model

        beta_best = []
        for i in range(len(ACTIONBEGIN)):
            features = state_to_features(new_game_state)
            beta_best.append(features@self.model[i])

        q_max = np.amax(beta_best)
        Y = R + gamma * q_max

        index = np.where(ACTIONS==self_action)[0]
        X = state_to_features(old_game_state)
        beta[index] = beta[index]+alpha*(X@(Y-X@beta[index]))
        
    
    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    beta = self.model
    index=np.where(ACTIONS==last_action)[0]
    X = state_to_features(last_game_state)
    Y = reward_from_events(self,events)

    beta[index]=beta[index]+0.1*(X@(Y-X@beta[index]))

    self.model = beta
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    
    game_rewards = {
        e.COIN_COLLECTED: 3,
        e.MOVED_UP: -0.2,
        e.MOVED_DOWN: -0.2,
        e.MOVED_LEFT: -0.2,
        e.MOVED_RIGHT: -0.2,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -5,
        CLOSER: 0.5,
        FURTHER: -0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
