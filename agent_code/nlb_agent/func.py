import numpy as np

#returns all the possible actions the agent could make
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
    if pos[2] == True:
        poss_moves.append('BOMB')
    #print('Debug: def possible_actions: executed succesfully')
    return poss_moves

#returns tuple (d,(x,y)): d is distance to next crate, (x,y) are the coordinates of the next crate. 
#This function is only called, when there are no coins on the field
def nearest_crate(game_state):
    #print('Debug: def nearesest_crate: called succesfully')

    field=game_state['field']
    #print(field)
    crates = np.argwhere(field==1)
    #print(crates)
    S=game_state['self']
    distances = np.array([])
    coo = []
    for crate in crates:
        x_dist= np.abs(crate[0]-S[3][0])
        y_dist= np.abs(crate[1]-S[3][1])
        distances = np.append(distances,x_dist+y_dist)
        coo.append(crate)
    
    index=np.argmin(distances)
    
    nearest_dist = distances[index]
    #print(nearest_dist)
    coordin = coo[index]

    #print('Debug: def nearest_crate: executed succesfully')
    return nearest_dist,coordin

#returns tupel (d,(x,y)): d is distance in taxicab geometry, (x,y) is the coordinate tupel
def nearest_coin(game_state): 
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

    if coo == []:
        return nearest_crate(game_state)
    index=np.argmin(distances)
    
    nearest_dist = distances[index]
    coordin = coo[index]

    #print('Debug: def nearest_coin: executed succesfully')
    return nearest_dist,coordin

#returns the number of destructible crates in the current state
def destroyable_crates(game_state):
    #print('Debug: des destroyable_crates called succesfully')
    field = game_state['field']
    pos = game_state['self'][3]
    #print('Pos:',pos)
    count = 0
    for i in range(-3,4):
        if pos[0]+i>15 or pos[0]<1:
            break
        else:
            #print('Position:', field[pos[0]])
            if field[pos[0]+i,pos[1]] == 1:
                count = count+1
        if pos[1]+i>15 or pos[1]<1:
            break
        else:
            if field[pos[0],pos[1]+i] == 1:
                count = count+1
    #print(count)
    #print('Debug: def destroyable_crates executed succesfully')
    return count

#it should return a tuple (d,(x,y)). d is the distance to the next safe spot, (x,y) are the coordinates
def safe_spot(game_state): 
    print('Debug: def safe_spot called succesfully')
    """Plan: make an array with all the possible threats and safe spots. 0=safe_spot, not0=possible threat"""
    S=game_state['self']        
    field = game_state['field']
    bombs = game_state['bombs']

    new_field=np.zeros((17,17),dtype=int)          
    new_field=new_field+game_state['explosion_map']
    new_field=new_field+field
    
    for bomb in bombs:
        x=bomb[0][0]
        y=bomb[0][1]
        value = bomb[1]
        new_field[bomb[0]]=value
        if new_field[x,y-1]==-1 and new_field[x-1,y] != -1 and new_field[x+1,y]!=-1 and new_field[x,y+1]!=-1:
            for i in range(1,4):
                new_field[x-i,y]

   
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

    #feature 4: number of destroyable crates
    num=destroyable_crates(game_state)
    features.append(num)

    #feature 5: distance to closest safe spot
    safe = safe_spot(game_state)[0]
    features.append(safe)

    #feature 6: horizontal distance to closest safe spot
    safecoordinates = safe_spot(game_state)[1]
    horisafe = safecoordinates[0]-state[3][0]
    features.append(horisafe)

    #feature 7: vertical distance to closest safe spot
    safecoordinates = safe_spot(game_state)[1]
    vertisafe = safecoordinates[1]-state[3][1]
    features.append(vertisafe)

    #print('Debug: state_to_features executed succesfully')
 
    return np.array(features)



    
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)

