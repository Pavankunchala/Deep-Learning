import numpy as np

import car

height = 4
width = 6
world = np.zeros((height, width))

# Define the initial car state
initial_position = [0, 0] # [y, x] (top-left corner)
velocity = [0, 1] # [vy, vx] (moving to the right)

carla = car.Car(initial_position, velocity, world)

print('Carla\'s initial state is: ' + str(carla.state))

carla.move()

# Track the change in state
print('Carla\'s state is: ' + str(carla.state))

# Display the world
carla.display_world()
