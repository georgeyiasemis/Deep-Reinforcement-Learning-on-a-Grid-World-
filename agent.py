import numpy as np
import torch
from deep_Qlearning import ReplayBuffer

# The Agent class allows the agent to interact with the environment.
class Agent():

    # The class initialisation function.
    def __init__(self, environment, dqn, target_net=None, gamma=0.9, buffer_maxlen=100000, batch_size=50):
        # Set the agent's environment.
        self.environment = environment
        # Set the agent's q-network
        self.dqn = dqn
        # Set the agent's target network
        self.target_net = target_net
        if self.target_net != None:
            # Init target network with dqn weights
            self.target_net.q_network.load_state_dict(self.dqn.q_network.state_dict())
        # Set the agent's replay buffer
        self.replaybuffer = ReplayBuffer(maxlen=buffer_maxlen)
        # Set the batch size. If it is equal to 1 it is the same as online learning.
        self.batch_size = batch_size
        # Set gamma for the Bellman Eq.
        self.gamma = gamma
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, epsilon=None, greedy=False):
        # Choose an action.
        if greedy:
            discrete_action = self.choose_greedy_action()
        else:
            # random if epsilon == None
            discrete_action = self.choose_epsilon_greedy_action(epsilon)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    def choose_random_action(self):

        return np.random.choice(3)

    def choose_greedy_action(self):
        state = torch.tensor(self.state)
        discrete_action = np.argmax(self.dqn.q_network(state).detach(), 0)
        return discrete_action.numpy().item()

    def choose_epsilon_greedy_action(self, epsilon):
        if epsilon == None:
            return self.choose_random_action()
        elif np.random.rand() < 1 - epsilon:
            # p(a = a*|s) = 1 - epsilon + epsilon / |A(s)|
            a = self.dqn.q_network(torch.tensor(self.state).float()).detach()
            return a.numpy().argmax().item()
        else:
            # p(a = a', a'!= a*|s) = epsilon / |A(s)|
            return self.choose_random_action()

    def get_greedy_trajectory(self, episode_step=20):
        self.reset()
        trajectory = [np.array(self.state).astype('float32')]
        for i in range(episode_step):
            self.step(greedy=True)
            trajectory.append(np.array(self.state).astype('float32'))
            if np.all(trajectory[-1] == self.environment.goal_state):
                return trajectory

        return trajectory

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = 1 - distance_to_goal
        return reward

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        # Up, Right, Down, Left
        actions = {0: np.array([0.1, 0]), 1: np.array([0, 0.1]),
                   2: np.array([-0.1, 0]), 3: np.array([0, -0.1])}

        return actions[discrete_action].astype('float32')
