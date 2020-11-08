def predict_state(state,dt):
    x= state[0]
    velocity = state[1]
    
    new_x = x+ velocity*dt
    
    
    predicted_state = [new_x,velocity]
    
    return predicted_state

inital_pos = 0
velocity = 50
inital_state = [inital_pos,velocity]

stateEST1 = predict_state(inital_state,2)

print("The state estimation after 2 seconds is:  "+ str(stateEST1))


    
    