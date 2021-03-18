import numpy as np
import torch
import torch.nn as nn
import collections
import random


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=200)
        self.output_layer = torch.nn.Linear(in_features=200, out_features=output_dimension)
        self.relu = torch.nn.ReLU()
    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).

    def forward(self, input):
        layer_1_output = self.relu(self.layer_1(input))
        layer_2_output = self.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN():

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        self.criterion = nn.MSELoss()

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition, gamma, target_net=None):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition, gamma, target_net)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, batch, gamma, target_net):
        # batch of size (N,) containing transitions (s, a, r, s')
        batch = batch.reshape(-1, 4)
        # Create an N x 2 tensor, where N = batch_size
        inp = torch.tensor([np.array(s) for s in batch[:,0]])
        # Forward to the network
        pred = self.q_network(inp)
        # Get Q-function for the corresponding actions
        actions = np.array(batch[:,1], dtype='int')
        pred = pred[range(pred.shape[0]), actions]

        # Create an N x 1 tensor containing the target values for each transition
        if target_net != None:
            # target = r + gamma * max_{a}(Q*^(s', a)) <-- from target net
            target = gamma * target_net.q_network(torch.tensor([np.array(s) for s in batch[:,-1]])).detach().max(1)[0]
        else:
            # target = r + gamma * max_{a}(Q(s', a)) <-- from q-network
            target = gamma * self.q_network(torch.tensor([np.array(s) for s in batch[:,-1]])).detach().max(1)[0]
        target += torch.tensor(np.array(batch[:,2], dtype='float')).float()
        # Calculate loss
        loss = self.criterion(pred, target)
        return loss

class ReplayBuffer():

    def __init__(self, maxlen=10000):

        self.deque = collections.deque(maxlen=maxlen)

    def append(self, transition):

        self.deque.append(transition)

    def sample(self, size):
        # Batch Size equal to 1 equivalent to online learning
        if size == 1:
            return np.array(self.deque[-1], dtype='object')
        return np.array(random.choices(self.deque, k=size), dtype='object')
