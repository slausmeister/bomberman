import numpy as np

def possible_actions(game_state):

    #print('Debug: def possible_actions: called succesfully')
    poss_moves = []
    pos = game_state['self']
    stats = game_state['field']

    if stats[pos[3][0],pos[3][1]+1] == 0:
        poss_moves.append('DOWN')
    if stats[pos[3][0],pos[3][1]-1] == 0:
        poss_moves.append('UP')
    if stats[pos[3][0]-1,pos[3][1]] == 0:
        poss_moves.append('LEFT')
    if stats[pos[3][0]+1,pos[3][1]] == 0:
        poss_moves.append('RIGHT')
    
    #if pos[2] == True:
    #    poss_moves.append('BOMB')
    
    #print('Debug: def possible_actions: executed succesfully')
    return poss_moves

def nearest_coin(game_state): #returns tupel (d,(x,y)): d is distance in taxicab geometry, (x,y) is the coordinate tupel
    #print('Debug: def nearest_coin: called succesfully')

    C=game_state['coins']
    S=game_state['self']
    distances = np.array([])
    coo = []
    for coins in C:
        x_dist= np.abs(coins[0]-S[3][0])
        y_dist= np.abs(coins[1]-S[3][1])
        distances = np.append(distances,x_dist+y_dist)
        coo.append(coins)

    index=np.argmin(distances)
    
    nearest_dist = distances[index]
    coordin = coo[index]

    #print('Debug: def nearest_coin: executed succesfully')
    return nearest_dist,coordin

    

def state_to_features(game_state: dict) -> np.array:

    #print('Debug: state_to_features called succesfully') 
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    state = game_state['self']

    features =[]
    #feature 1: total distance to closest coin
    dist=nearest_coin(game_state)[0]
    coordinates = nearest_coin(game_state)[1]
    features.append(dist)

    #feature 2: horizontal distance to closest coin
    hori=coordinates[0]-state[3][0]
    #if hori >= 0:
    #    print('Debug: coin is to the right')
    #else:
    #    print('Debug: coin is to the left')
    features.append(hori)

    #feature 3: vertical distance to closest coin
    vert=coordinates[1]-state[3][1]
    #if vert >=0:
    #    print('Debug: coin is below')
    #else:
    #    print('Debug: coin is above')
    features.append(vert)

    #print('Debug: state_to_features executed succesfully')
    
    return np.array(features)
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)


