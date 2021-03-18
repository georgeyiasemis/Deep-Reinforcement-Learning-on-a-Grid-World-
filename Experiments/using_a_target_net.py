import sys
sys.path.insert(1, '../')

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
# Import the environment module1
from environment import Environment
# Import agent and DQN
from agent import Agent
from deep_Qlearning import DQN


# =============================================================================
# Using a deep-Target Network (DTN)
# =============================================================================

# seed = 0
# np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

# Without a DTN
environment_1 = Environment(display=False, magnification=500, id=1)

# With a DTN
environment_2 = Environment(display=False, magnification=500, id=2)


gamma = 0.7
batch_size = 100
epsilon = None

# Create two agents
agent1 = Agent(environment_1, DQN(), gamma=gamma, batch_size=batch_size)
agent2 = Agent(environment_2, DQN(), target_net=DQN(), gamma=gamma, batch_size=batch_size)
update_target_every_steps = 20

# Number of episodes
num_episodes = 100
num_episode_steps = 20
episodes_count, steps_count = 0, 0

# Plot loss for each case
losses = [[], []]

plot_greedy_trajectory = True
plot_Q_fun = True

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
        transition2 = agent2.step(epsilon=epsilon)
        agent1.replaybuffer.append(transition1)
        agent2.replaybuffer.append(transition2)

        # Do not start training until the ReplayBuffer has stored at least batch_size samples
        if steps_count >= batch_size:
            # Update target network weights
            if steps_count % update_target_every_steps == 0:
                agent2.target_net.q_network.load_state_dict(agent2.dqn.q_network.state_dict())

            # Sample a batch from the Replay Buffer
            batch1 = agent1.replaybuffer.sample(size=agent1.batch_size)
            batch2 = agent2.replaybuffer.sample(size=agent2.batch_size)
            # Store loss
            losses[0].append(agent1.dqn.train_q_network(batch1, agent1.gamma))
            losses[1].append(agent2.dqn.train_q_network(batch2, agent2.gamma, agent2.target_net))

    # Visualise Q-function
    if plot_Q_fun:
        Qs = agent1.dqn.q_network(agent1.environment.states).detach()
        agent1.environment.plot_Qs(Qs)
        Qs = agent2.dqn.q_network(agent2.environment.states).detach()
        agent2.environment.plot_Qs(Qs)

    # Visualise greedy trajectory after each episode
    if plot_greedy_trajectory:
        greedy_trajectory1 = agent1.get_greedy_trajectory()
        agent1.environment.plot_trace(greedy_trajectory1)
        greedy_trajectory2 = agent2.get_greedy_trajectory()
        agent2.environment.plot_trace(greedy_trajectory2)

# Plot loss for every step
plt.figure(figsize=(10,8))
plt.semilogy(range(agent1.batch_size, agent1.batch_size + len(losses[0]) ), losses[0], label='Deep Q Learning')
plt.semilogy(range(agent2.batch_size, agent2.batch_size + len(losses[1]) ), losses[1], label='Deep Q Learning with a target network')
plt.xlabel('Steps'); plt.ylabel('MSE Loss'); plt.title('Loss function')
plt.legend()
plt.show()
