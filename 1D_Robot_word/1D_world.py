import matplotlib.pyplot as plt
import numpy as np

# uniform distribution for 5 grid cells
p = [0.2, 0.2, 0.2, 0.2, 0.2]
print(p)

#display map
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
        
display_map(p)

#writing a function that intializes 1D robot and returns a probablity distribution of robot in each space

def initalize_robot(grid_length):
    ''' Takes in a grid length and returns 
    a uniform distribution of location probabilities'''
    
    p = []
    # create a list that has the value of 1/grid_length for each cell
    for i in range(grid_length):
        
        p.append(1.0/grid_length)
    return p

#initalizing it again
p = initalize_robot(8)
print(p)
display_map(p, bar_width=0.9)

    
       
    

