import numpy as np
import torch
import matplotlib.pyplot as plt
import random
# Import the environment module1
from environment import Environment
# Import agent and DQN
from starter_code import Agent, DQN

# Using a deep-Target Network (DTN)

seed = 0
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)


# Without a DTN
environment_1 = Environment(display=False, magnification=500, id=1)

# With a DTN
environment_2 = Environment(display=False, magnification=500, id=2)


gamma = 0.7
batch_size = 100

# Create two agents
agent1 = Agent(environment_1, DQN(), gamma=gamma, batch_size=batch_size)
agent2 = Agent(environment_2, DQN(), target_net=DQN(), gamma=gamma, batch_size=batch_size)

# Number of episodes
num_episodes = 100
num_episode_steps = 20
episodes_count, steps_count = 0, 0

# Plot loss for each case
losses = [[], []]

plot_greedy_trajectory = True
plot_Q_fun = False

# Loop over episodes
while episodes_count < num_episodes:
    # Reset the environment for the start of the episode.
    agent1.reset()
    agent2.reset()
    episodes_count += 1
    # Loop over steps within this episode. The episode length here is 20.
    for step_num in range(num_episode_steps):
        steps_count += 1
        # Step the agent once, and get the transition tuple for this step
        transition1 = agent1.step(epsilon=epsilon)
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
