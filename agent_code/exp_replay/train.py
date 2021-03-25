from agent_code.exp_replay.func import *
import pickle
import random
import numpy as np
import sys
from collections import namedtuple, deque
from typing import List
import copy as copy

import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

CLOSERCOIN = 'CLOSERCOIN'
FURTHERCOIN = 'FURTHERCOIN'
CLOSERCRATE = 'CLOSERCRATE'
FURTHERCRATE = 'FURTHERCRATE'
SAFE = 'SAFE'
NOTSAFE = 'NOTSAFE'
NICEBOMB = 'NICEBOMB'
CLOSERSAFE = 'CLOSERSAFE'
FURTHERSAFE = 'FURTHERSAFE'



# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
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
        if safe_spot(old_game_state)[1] == new_game_state['self'][3]:
            events.append(SAFE)
            print("SAFE")
        if free_space(new_game_state, explosions=True)[new_game_state['self'][3]] == False:
            events.append(NOTSAFE)
            print("NOT SAFE")
        if destroyable_crates(old_game_state) >=2 and new_game_state['self'][2]==0:
            events.append(NICEBOMB)

        R = reward_from_events(self,events)
        X = state_to_features(old_game_state)
        X_prime = state_to_features(new_game_state)
        index = np.where(ACTIONS==self_action)[0][0]

        self.logger.info(f"Reward: {R} from move {self_action}")
        
        
    # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    # (s, a, r, s')
    
        self.transitions.appendleft((X,self_action,R,X_prime))

        if len(self.transitions)==TRANSITION_HISTORY_SIZE:
            exps = random.sample(self.transitions, 6)
            for exp in exps:
                train(exp,self)
        

def train(experience,self):
    beta_current = copy.deepcopy(self.model)

    beta_prime_best = []
    for i in range(len(ACTIONS)):
        beta_prime_best.append((experience[0]@beta_current[i],i))


    beta_prime_best.sort(key=lambda x: x[0], reverse=True)
    i=beta_prime_best[0][1]
    
    self.model[i] = beta_current[i] + 0.001*(experience[2] + (gamma * beta_prime_best[0][0]))
    #print(beta)
















def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')d
    exps = random.sample(self.transitions, len(self.transitions))
    for exp in exps:
        train(exp,self)
    

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
        e.KILLED_SELF: -10,
        e.INVALID_ACTION: -5,
        CLOSERCOIN: 1,
        FURTHERCOIN: -1,
        CLOSERCRATE: 0.3,
        FURTHERCRATE: -0.3,
        CLOSERSAFE: 1,
        FURTHERSAFE: -1,
        SAFE: 3,
        NOTSAFE: -1,
        NICEBOMB: 0

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print(reward_sum)
    return reward_sum
