import numpy as np

def taxi(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])
 
def build_state(self,game_state):


    #Position of player: state[0]
    state = np.zeros((17,17))
    state[game_state['self'][3]]=1
    

    #nearest coin: state[1]
    state=np.stack((state,np.zeros((17,17))))
    dist = taxi(game_state['coins'][0], game_state['self'][3])
    state[1][game_state['coins'][0]]=1
    for i in game_state['coins']:
        #state[3][i]=1
        if taxi(i, game_state['self'][3]) < dist:
            dist = taxi(i, game_state['self'][3])
            state[1].fill(0)
            state[1][i]=1


    #rotate and inticate current quadrant: state[2]
    if game_state['self'][3][0]>7:
        if game_state['self'][3][1]>7:
            state = np.rot90(state,2, axes=(1,2))
            quad="rd"
            self.order_rd = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
            
        else:
            state = np.rot90(state,3, axes=(1,2))
            quad="ru"
            self.order_ru = {"LEFT": "UP", "RIGHT": "DOWN", "UP": "RIGHT", "DOWN": "LEFT"}
    else: 
        if game_state['self'][3][1]>7:
            state = np.rot90(state,1, axes=(1,2))
            quad="ld"
            self.order_ld = {"LEFT": "DOWN", "UP": "LEFT", "RIGHT": "UP", "DOWN": "RIGHT"}
        else:
            quad="lu"        

    return state, quad