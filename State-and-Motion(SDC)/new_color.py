import numpy as np
import car

height = 4
width = 6
world = np.zeros((height, width))

initial_position = [0, 0] # [y, x] (top-left corner)
velocity = [0, 1] # [vy, vx] (moving to the right)

carla = car.Car(initial_position, velocity, world)

position2 = [2,2]
velocity2 = [1,0]

jean = car.Car(position2,velocity2,world,'y')

carla.move()
carla.move()
carla.move()
carla.turn_left()
carla.move()
carla.display_world()

#for jean
jean.move()
jean.move()
jean.move()
jean.turn_left()
jean.move()
jean.display_world()

