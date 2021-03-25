from agent_code.nlb_agent.func import destroyable_crates, nearest_coin, safe_spot
import pickle
import random
import numpy as np
import sys
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

CLOSERCOIN = 'CLOSERCOIN'
FURTHERCOIN = 'FURTHERCOIN'
CLOSERCRATE = 'CLOSERCRATE'
FURTHERCRATE = 'FURTHERCRATE'
CLOSERSAFE = 'CLOSERSAFE'
FURTHERSAFE = 'FURTHERSAFE'
NICEBOMB = 'NICEBOMB'


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

alpha=0.001
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
            events.append(CLOSERCOIN)
        if nearest_coin(old_game_state)[0] < nearest_coin(new_game_state)[0]:
            events.append(FURTHERCOIN)
        if destroyable_crates(old_game_state) < destroyable_crates(new_game_state):
            events.append(CLOSERCRATE)
        if destroyable_crates(old_game_state) > destroyable_crates(new_game_state):
            events.append(FURTHERCRATE)
        if safe_spot(old_game_state)[0] > safe_spot(new_game_state)[0]:
            events.append(CLOSERSAFE)
        if safe_spot(old_game_state)[0] < safe_spot(new_game_state)[0]:
            events.append(FURTHERSAFE)
        if destroyable_crates(old_game_state) >=2 and new_game_state['self'][2]==0:
            events.append(NICEBOMB)

        R = reward_from_events(self,events)
        X = state_to_features(old_game_state)
        index = np.where(ACTIONS==self_action)[0][0]
        
        beta = self.model
        
        beta_best = []
        for i in range(len(ACTIONS)):
            features = state_to_features(new_game_state)
            beta_best.append(features@beta[i])

        q_max = np.amax(beta_best)
        delta = R + gamma * q_max - X@beta[index]
        
        #print(new_game_state['round'])
        for i in range(len(beta[index])):
            beta[index][i] = beta[index][i]+alpha*delta*state_to_features(old_game_state)[i]
            if beta[index][i]>=1000:
                print('Runde:', new_game_state['round'])
                sys.exit('Zahlen zu groß')
        #print(beta)
        self.model=beta
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
    index=np.where(ACTIONS==last_action)[0][0]
    X = state_to_features(last_game_state)
    R= reward_from_events(self,events)
    beta_best = []
    for i in range(len(ACTIONS)):
         features = state_to_features(last_game_state)
         beta_best.append(features@beta[i])

    q_max = np.amax(beta_best)
    delta = R + gamma * q_max - X@beta[index]
    for i in range(len(beta[index])):
        beta[index][i] = beta[index][i]+alpha*delta*state_to_features(last_game_state)[i]
    
    #print(beta[1])
    self.model = beta
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.MOVED_UP: -.1,
        e.MOVED_DOWN: -.1,
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.WAITED: -0.1,
        e.CRATE_DESTROYED: 3,
        e.KILLED_SELF: -5,
        #e.INVALID_ACTION: -5,
        CLOSERCOIN: 0.5,
        FURTHERCOIN: -0.5,
        CLOSERCRATE: 0.3,
        FURTHERCRATE: -0.3,
        CLOSERSAFE: 2,
        FURTHERSAFE: -1,
        NICEBOMB: 3

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print(reward_sum)
    return reward_sum
