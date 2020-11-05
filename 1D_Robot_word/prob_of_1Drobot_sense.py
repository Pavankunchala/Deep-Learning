import matplotlib.pyplot as plt
import numpy as np


def initalize_robot(grid_length):
    ''' Takes in a grid length and returns 
    a uniform distribution of location probabilities'''
    
    p = []
    # create a list that has the value of 1/grid_length for each cell
    for i in range(grid_length):
        
        p.append(1.0/grid_length)
    return p

def display_map(grid,bar_width=1):
    if(len(grid)>0):
        x_labels = range(len(grid))
        plt.bar(x_labels, height=grid, width=bar_width, color='b')
        plt.xlabel('Grid Cell')
        plt.ylabel('Probability')
        plt.ylim(0, 1) # range of 0-1 for probability values 
        plt.title('Probability of the robot being at each cell in the grid')
        plt.xticks(np.arange(min(x_labels), max(x_labels)+1, 1))
        plt.show()
        
    else:
        print("Grid is Empty")
        
p = initalize_robot(5)
pHit  = 0.6
pMiss = 0.2

# Creates a new grid, with modified probabilities, after sensing
# All values are calculated by a product of 1. the sensing probability for a color (pHit for red)
# and 2. the current probability of a robot being in that location p[i]; all equal to 0.2 at first.
p[0] = p[0]*pMiss
p[1] = p[1]*pHit
p[2] = p[2]*pHit
p[3] = p[3]*pMiss
p[4] = p[4]*pMiss

print(p)
display_map(p,bar_width=0.9)