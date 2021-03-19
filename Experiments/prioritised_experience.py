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
# Prioritised Experience Replay Buffer
# =============================================================================

seed = 1
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)


environment_1 = Environment(display=False, magnification=500, id=1)
environment_2 = Environment(display=False, magnification=500, id=2)


# R(s) = (1 - dist(s, goal))^2
rew_fun = reward_fun_a(a=2)

gamma = 0.7
batch_size = 25

# For agent 2
buffer = PrioritisedExperienceReplayBuffer(maxlen=10000, epsilon=0.01, alpha=3)

# Create agents
agent_1 = Agent(environment_1, DQN(), target_net=DQN(), gamma=gamma, batch_size=batch_size, reward_fun=rew_fun)
agent_2 = Agent(environment_2, DQN(), target_net=DQN(), gamma=gamma, batch_size=batch_size, replay_buffer=buffer, reward_fun=rew_fun)


# Policy arameters
starting_epsilon = 1.0
epsilon = None
delta = 0.6
update_target_every_steps = 20

# Number of episodes
num_episodes = 100
num_episode_steps = 25
episodes_count, steps_count = 0, 0

losses = [[], []]
# Plot loss for each case

plot_greedy_trajectory = True
plot_Q_fun = True

# Loop over episodes
while episodes_count < num_episodes:
    # Reset the environment for the start of the episode.
    agent_1.reset()
    agent_2.reset()
    episodes_count += 1
    # Loop over steps within this episode. The episode length here is 20.
    for step_num in range(num_episode_steps):
        steps_count += 1
        transition_1 = agent_1.step(epsilon=epsilon)
        agent_1.replaybuffer.append(transition_1)
        transition_2 = agent_2.step(epsilon=epsilon)
        agent_2.replaybuffer.append(transition_2)

        # Do not use e-greedy policy before training starts
        if steps_count == batch_size - 1:
            epsilon = starting_epsilon

        # Do not start training until the ReplayBuffer has stored at least batch_size samples
        if steps_count >= batch_size:
            # Update target network weights
            if steps_count % update_target_every_steps == 0:
                agent_1.target_net.q_network.load_state_dict(agent_1.dqn.q_network.state_dict())
                agent_2.target_net.q_network.load_state_dict(agent_2.dqn.q_network.state_dict())
                if epsilon != None:
                # Decay epsilon by delta
                    epsilon = max(0.05, epsilon * delta)
            batch_1 = agent_1.replaybuffer.sample(batch_size)
            batch_2, ind_2 = agent_2.replaybuffer.sample(batch_size)

            loss_1 = agent_1.dqn.train_q_network(batch_1, agent_1.gamma)[0]
            losses[0].append(loss_1)

            loss_2, td_errors_2 = agent_2.dqn.train_q_network(batch_1, agent_2.gamma)
            agent_2.replaybuffer.weight_update(td_errors_2, ind_2)
            losses[1].append(loss_2)

    # Visualise greedy trajectory after each episode
    greedy_trajectory_1 = agent_1.get_greedy_trajectory()
    agent_1.environment.plot_trace(greedy_trajectory_1)

    greedy_trajectory_2 = agent_2.get_greedy_trajectory()
    agent_2.environment.plot_trace(greedy_trajectory_1)


plt.figure(figsize=(10,8))
plt.plot(losses[0], label='Not Prioritised Experience')
plt.plot(losses[1], label='Prioritised Experience')
plt.xlabel('Steps'); plt.ylabel('Loss Function');
plt.legend()
plt.show()
