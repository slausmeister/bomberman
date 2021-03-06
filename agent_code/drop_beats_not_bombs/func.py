import numpy as np
from operator import itemgetter

def norm(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

#returns all the possible actions the agent could make
def possible_actions(self,game_state):
    reduced=['UP','DOWN','LEFT','RIGHT']
    #print('Debug: def possible_actions: called succesfully')
    poss_moves = []
    pos = game_state['self']
    stats = game_state['field']
    
    if self.history.count(pos[3])>2:
        return np.random.choice(reduced,1)
    else:
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
    
    #to avoid bug
    if coo == []:
        return 0,(0,0)
    index=np.argmin(distances)
    
    nearest_dist = distances[index]
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

    #if no coin can be found, reward the agent for getting closer to a crate
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
    count = 0
    #temp = np.zeros((21,21))
    if pos[0]%2 == 0:
        for i in range(-3,4):
            if pos[0]+i>15 or pos[0]<1:
                break
            else:
                if field[pos[0]+i,pos[1]] == 1:
                    count = count+1
    if pos[1]%2 == 0:
        for i in range(-3,4):
            if pos[1]+i>15 or pos[1]<1:
                break
            else:
                if field[pos[0],pos[1]+i] == 1:
                    count = count+1
    #print(count)
    #print('Debug: def destroyable_crates executed succesfully')
    return count

    """def free_space(game_state, explosions = True):
    # A function that returns an array indicating free spots on the grid.
    # if explosions == False, exposions are not counted as an obstacle
    # Returns an array with True for free spots

    S=game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']

    new_field=np.zeros((17,17),dtype=float)          
    new_field+=0.3*game_state['explosion_map']
    new_field+=0.5*field # Adding half, to prevent bombs canceling crates, thus making the cell appear free
    # Initialising a larger array containing future explosions. The array will be cut to (15,15), as that is the relevant grid for explosions
    temp = np.zeros((21,21))
    for bomb in bombs :
        x=bomb[0][0]+2
        y=bomb[0][1]+2
        temp[(x,y)]=5

        if explosions == True:
            #Checking wether a wall will block the explosion
            if x%2!=0:
                for i in [-3,-2,-1,1,2,3]:
                    temp[(x,y+i)]=5
            if y%2!=0:
                for i in [-3,-2,-1,1,2,3]:
                    temp[(x+i,y)]=5
    # slicing the array to size
    temp = temp[2:19,2:19]
    new_field += temp

    # Building boolean array
    boolean = np.ones((17,17), dtype=bool)

    boolean[np.nonzero(new_field)] = False
    return boolean
 
 
    def look_for_targets(game_state,free_space, start, targets):

    if len(targets) == 0: return nearest_crate(game_state)

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min() #Taxi to closest target

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist: #Right way?
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    # Determine the first step towards the best found target tile
    current = best
    iteration = 0
    while True:
        iteration += 1
        if parent_dict[current] == start: 
            return (iteration,tuple(current))
        current = parent_dict[current]""" 

#it should return a tuple (d,(x,y)). d is the distance to the next safe spot, (x,y) are the coordinates
def safe_spot(game_state): 
    #print('Debug: def safe_spot called successfully')
    """Plan: make an array with all the possible threats and safe spots. 0=safe_spot, not0=possible threat"""
    S=game_state['self']        
    field = game_state['field']
    bombs = game_state['bombs']

    new_field=np.zeros((17,17),dtype=float)          
    new_field+=0.3*game_state['explosion_map']
    new_field+=0.5*field # Adding half, to prevent bombs canceling crates, thus making the cell appear free
    # Initialising a larger array containing future explosions. The array will be cut to (15,15), as that is the relevant grid for explosions
    temp = np.zeros((21,21))
    for bomb in bombs :
        x=bomb[0][0]+2
        y=bomb[0][1]+2
        temp[(x,y)]=5
        #Checking wether a wall will block the explosion
        if x%2!=0:
            for i in [-3,-2,-1,1,2,3]:
                temp[(x,y+i)]=5
        if y%2!=0:
            for i in [-3,-2,-1,1,2,3]:
                temp[(x+i,y)]=5
    # slicing the array to size
    temp = temp[2:19,2:19]
    new_field += temp


    # We will now iterate through the array to find the closest safe spot
    temp = []
    for i in np.argwhere(new_field == 0):
        temp.append(i)


    safe_spots = []
    for i in temp:
        dist=norm(i, game_state['self'][3])
        safe_spots.append((dist,tuple(i)))
    
    safe_spots.sort(key=lambda x: x[0::])

    #print('Debug: safe_spot executed successfully')
    return safe_spots[0]

   
def state_to_features(game_state: dict) -> np.array:
    #print('Debug: state_to_features called succesfully') 

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    state = game_state['self']

    features =[]
    #feature 1: total distance to closest coin
    if nearest_coin(game_state)[0]==0:
        dist=0
        hori=0
        vert=0
    else:
        dist=nearest_coin(game_state)[0]
        coordinates = nearest_coin(game_state)[1]
        #feature 2: horizontal distance to closest coin
        hori=coordinates[0]-state[3][0]
        #feature 3: vertical distance to closest coin
        vert=coordinates[1]-state[3][1]
        
    features.append(dist)
    features.append(hori)
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


