import sys
sys.path.insert(1, '../')

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
# Import the environment module1
from environment import Environment
# Import agent and DQN
from agent import *
from deep_Qlearning import DQN

# =============================================================================
# ONLINE LEARNING (batch size = 1) vs BATCH LEARNING
# =============================================================================

seed = 0
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

# Online Learning
environment_1 = Environment(display=False, magnification=500, id=1)
batch_size_1 = 1

# Batch learning
environment_2 = Environment(display=False, magnification=500, id=2)
batch_size_2 = 50

gamma = 0.7

# Create two agents
agent1 = Agent(environment_1, DQN(), gamma=gamma, batch_size=batch_size_1)
agent2 = Agent(environment_2, DQN(), gamma=gamma, batch_size=batch_size_2)

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
        transition1, transition2 = agent1.step(), agent2.step()
        # trace.append(agent.state())
        agent1.replaybuffer.append(transition1)
        agent2.replaybuffer.append(transition2)


        # Get last transition only
        transition = agent1.replaybuffer.sample(size=agent1.batch_size)
        # Store loss
        loss = agent1.dqn.train_q_network(transition, agent1.gamma)[0]
        losses[0].append(loss)


        if steps_count >= agent2.batch_size:

            # Sample a batch from the Replay Buffer
            batch = agent2.replaybuffer.sample(size=agent2.batch_size)
            # Store loss
            loss = agent2.dqn.train_q_network(batch, agent2.gamma)[0]
            losses[1].append(loss)

        if plot_Q_fun:
            Qs = agent1.dqn.q_network(agent1.environment.states).detach()
            agent1.environment.plot_Qs(Qs)
            Qs = agent2.dqn.q_network(agent2.environment.states).detach()
            agent2.environment.plot_Qs(Qs)

    # Plot greedy trajectory after each episode
    if plot_greedy_trajectory:
        greedy_trajectory1 = agent1.get_greedy_trajectory()
        agent1.environment.plot_trace(greedy_trajectory1)
        greedy_trajectory2 = agent2.get_greedy_trajectory()
        agent2.environment.plot_trace(greedy_trajectory2)

# Plot loss for every step
plt.figure(figsize=(10,8))
plt.semilogy(range(agent1.batch_size, agent1.batch_size + len(losses[0]) ), losses[0], label='Online Learning')
plt.semilogy(range(agent2.batch_size, agent2.batch_size + len(losses[1]) ), losses[1], label='Batch Learning')
plt.xlabel('Steps'); plt.ylabel('MSE Loss'); plt.title('Loss function')
plt.legend()
plt.show()
