import matplotlib.pyplot as plt

class Car(object):
    def __init__(self,position,velocity,world):
        self.state = [position,velocity]
        self.world = world # world is a 2D list of values that range from 0-1
        
        self.color = 'r'
        self.path = []
        self.path.append(position)
        
    def move(self,dt=1):
        height = len(self.world)
        width = len(self.world[0])
        
        position = self.state[0]
        velocity = self.state[1]
        
        
        predicted_position = [
            (position[0]+ velocity[0]* dt)% height,
            (position[1]+ velocity[1]* dt)% width
        ]
        
        self.state = [predicted_position, velocity]
        
        self.path.append(predicted_position)
        
    def turn_left(self):
        velocity = self.state[1]
        
        predicted_velocity = [
            -velocity[1],
            velocity[0]
        ]
        self.state[1] = predicted_velocity
        
    def display_world(self):
        # Store the current position of the car
        position = self.state[0]
        
        # Plot grid of values + initial ticks
        plt.matshow(self.world, cmap='gray')
        
        # Set minor axes in between the labels
        ax=plt.gca()
        rows = len(self.world)
        cols = len(self.world[0])

        ax.set_xticks([x-0.5 for x in range(1,cols)],minor=True )
        ax.set_yticks([y-0.5 for y in range(1,rows)],minor=True)

        # Plot grid on minor axes in gray (width = 2)
        plt.grid(which='minor',ls='-',lw=2, color='gray')
        
        # Create a 'x' character that represents the car
        # ha = horizontal alignment, va = verical
        ax.text(position[1], position[0], 'x', ha='center', va='center', color=self.color, fontsize=30)
            
        # Draw path if it exists
        if(len(self.path) > 1):
            # loop through all path indices and draw a dot (unless it's at the car's location)
            for pos in self.path:
                if(pos != position):
                    ax.text(pos[1], pos[0], '.', ha='center', va='baseline', color=self.color, fontsize=30)

        # Display final result
        plt.show()

