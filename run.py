import numpy as np
import torch
import matplotlib.pyplot as plt


# Import the environment module1
from environment import Environment
# Import agent and DQN
from starter_code import Agent, DQN

# seed = 0
# np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

# Create an environment.
# If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
# Magnification determines how big the window will be when displaying the environment on your monitor. For desktop PCs, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
environment = Environment(display=False, magnification=500)
# Create a DQN (Deep Q-Network) & a target network
dqn = DQN()
target_net = DQN()

# Batch Size
batch_size = 50

# Create an agent
agent = Agent(environment, dqn, target_net, batch_size=batch_size)

# Policy arameters
starting_epsilon = None
epsilon = None
# Decay factor
delta = 0.99
# Target network parameters
# Set to 1 to not use a target network
update_target_every_steps = 20

# Number of episodes
num_episodes = 25
episodes_count, steps_count = 0, 0

# Visualisations
plot_Q_fun = True
plot_greedy_trajectory = True

# Store losses
losses = []

# Loop over episodes
while episodes_count < num_episodes:
    # Reset the environment for the start of the episode.
    agent.reset()
    episodes_count += 1
    # Loop over steps within this episode. The episode length here is 20.
    for step_num in range(20):
        steps_count += 1
        # Step the agent once, and get the transition tuple for this step
        transition = agent.step(epsilon=epsilon)
        # trace.append(agent.state())
        agent.replaybuffer.append(transition)
        # Do not use e-greedy policy before training starts
        if steps_count == agent.batch_size - 1:
            epsilon = starting_epsilon
        # Do not start training until the ReplayBuffer has stored at least batch_size samples
        if steps_count >= agent.batch_size:
            # Update target network weights
            if steps_count % update_target_every_steps == 0:
                agent.target_net.q_network.load_state_dict(agent.dqn.q_network.state_dict())
            # Decay epsilon
            if epsilon != None:
                epsilon = max(0.05, abs(epsilon * delta)) 
            # Sample a batch from the Replay Buffer
            batch = agent.replaybuffer.sample(size=agent.batch_size)
            # Store loss
            loss = agent.dqn.train_q_network(batch, agent.gamma, agent.target_net)
            losses.append(loss)
            # Plot Q-function animation
            if plot_Q_fun:
                Qs = agent.dqn.q_network(agent.environment.states).detach()
                agent.environment.plot_Qs(Qs)

    # Plot greedy trajectory after each episode
    if plot_greedy_trajectory:
        greedy_trajectory = agent.get_greedy_trajectory()
        agent.environment.plot_trace(greedy_trajectory)

# Plot loss for every step
plt.semilogy(range(agent.batch_size, agent.batch_size + len(losses) ), losses)
plt.xlabel('Steps'); plt.ylabel('MSE Loss'); plt.title('Loss function')
