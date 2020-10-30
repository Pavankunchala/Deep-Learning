import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import  Variable

# so that random variables will be consistent and repeatable for testing
torch.manual_seed(2)

#an LSTM with 4 inputs and 3  hidden dim
# this expects to see 4 values as input and generates 3 values as output

input_dim = 4
hidden_dim = 3
lstm = nn.LSTM(input_size  = input_dim,hidden_size = hidden_dim)

# make 5 input sequences of 4 random values each
inputs_list = [torch.randn(1, input_dim) for _ in range(5)]
print('inputs: \n', inputs_list)
print('\n')

# initialize the hidden state
# (1 layer, 1 batch_size, 3 outputs)
# first tensor is the hidden state, h0
# second tensor initializes the cell memory, c0
h0 = torch.randn(1, 1, hidden_dim)
c0 = torch.randn(1, 1, hidden_dim)


h0 = Variable(h0)
c0 = Variable(c0)
# step through the sequence one element at a time.
for i in inputs_list:
    # wrap in Variable 
    i = Variable(i)
    
    # after each step, hidden contains the hidden state
    out, hidden = lstm(i.view(1, 1, -1), (h0, c0))
    print('out: \n', out)
    print('hidden: \n', hidden)

###
# turn inputs into a tensor with 5 rows of data
# add the extra 2nd dimension (1) for batch_size
inputs = torch.cat(inputs_list).view(len(inputs_list), 1, -1)

# print out our inputs and their shape
# you should see (number of sequences, batch size, input_dim)
print('inputs size: \n', inputs.size())
print('\n')

print('inputs: \n', inputs)
print('\n')

# initialize the hidden state
h0 = torch.randn(1, 1, hidden_dim)
c0 = torch.randn(1, 1, hidden_dim)

# wrap everything in Variable
inputs = Variable(inputs)
h0 = Variable(h0)
c0 = Variable(c0)
# get the outputs and hidden state
out, hidden = lstm(inputs, (h0, c0))

print('out: \n', out)
print('hidden: \n', hidden)