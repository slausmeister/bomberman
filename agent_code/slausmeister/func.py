import numpy as np

def taxi(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def nearest_coin(game_state):

    dist = taxi(game_state['coins'][0], game_state['self'][3])
    close_coin = (dist,(game_state['coins'][0]))
    for i in game_state['coins']:
        dist_temp=taxi(i, game_state['self'][3])
        if dist_temp < close_coin[0]:
            close_coin = (dist_temp,(game_state['coins'][0]))
    return close_coin

 
def oriented_state(self, game_state):


    state = np.zeros((17,17))
    state[game_state['self'][3]]=1
    

    #Additng stuff to rotation
    state=np.stack((state,np.zeros((17,17))))
    close_coin=nearest_coin(game_state)
    state[1][close_coin[1]]=1


    #rotate and inticate current quadrant: state[2]
    if game_state['self'][3][0]>7:
        if game_state['self'][3][1]>7:
            state = np.rot90(state,2, axes=(1,2))
            quad="rd"
            #self.order_rd = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
            
        else:
            state = np.rot90(state,3, axes=(1,2))
            quad="ru"
            #self.order_ru = {"LEFT": "UP", "RIGHT": "DOWN", "UP": "RIGHT", "DOWN": "LEFT"}
    else: 
        if game_state['self'][3][1]>7:
            state = np.rot90(state,1, axes=(1,2))
            quad="ld"
            #self.order_ld = {"LEFT": "DOWN", "UP": "LEFT", "RIGHT": "UP", "DOWN": "RIGHT"}
        else:
            quad="lu"        

    return state, quad

def state_identification(state):
    # The [1:-1] is to remove the square brakets
    return str(np.nonzero(state[0])[0])[1:-1] + str(np.nonzero(state[0])[1])[1:-1] + str(np.nonzero(state[1])[0])[1:-1] + str(np.nonzero(state[1])[1])[1:-1]



