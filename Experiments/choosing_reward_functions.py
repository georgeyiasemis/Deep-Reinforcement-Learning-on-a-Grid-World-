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
# Experimenting with different reward functions
# =============================================================================

# seed = 0
# np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

environment_1 = Environment(display=False, magnification=500, id=1)
environment_2 = Environment(display=False, magnification=500, id=2)
environment_3 = Environment(display=False, magnification=500, id=3)

# R1(s) = 1 - dist(s, goal)
rew_fun_1 = reward_fun_a(a=1)
# R2(s) = (1 - dist(s, goal))^2
rew_fun_2 = reward_fun_a(a=1)
# R3(s) = 1 if s==goal, -1 otherwise
rew_fun_3 = step_reward_fun()



gamma = 0.7
batch_size = 100
epsilon = None

# Create agents
agent1 = Agent(environment_1, DQN(), target_net=DQN(), gamma=gamma, batch_size=batch_size, reward_fun=rew_fun_1)
agent2 = Agent(environment_2, DQN(), target_net=DQN(), gamma=gamma, batch_size=batch_size, reward_fun=rew_fun_2)
agent3 = Agent(environment_3, DQN(), target_net=DQN(), gamma=gamma, batch_size=batch_size, reward_fun=rew_fun_3)

agents = [agent1, agent2, agent3]

# Policy arameters
starting_epsilon = 1.0
epsilon = None
delta = 0.6
update_target_every_steps = 20

# Number of episodes
num_episodes = 25
num_episode_steps = 20
episodes_count, steps_count = 0, 0

# Plot loss for each case

plot_greedy_trajectory = True
plot_Q_fun = True

distances = [[], [], []]
steps = [num_episode_steps * i for i in range(1, num_episodes+1)]

# Loop over episodes
while episodes_count < num_episodes:
    # Reset the environment for the start of the episode.
    for agent in agents:
        agent.reset()
    episodes_count += 1
    # Loop over steps within this episode. The episode length here is 20.
    for step_num in range(num_episode_steps):
        steps_count += 1
        for agent in agents:
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step(epsilon=epsilon)
            agent.replaybuffer.append(transition)
            
            # Do not use e-greedy policy before training starts
            if steps_count == agent.batch_size - 1:
                epsilon = starting_epsilon
            
        # Do not start training until the ReplayBuffer has stored at least batch_size samples
        if steps_count >= batch_size:
            # Update target network weights
            if steps_count % update_target_every_steps == 0:
                for agent in agents:
                    agent.target_net.q_network.load_state_dict(agent.dqn.q_network.state_dict())
            if epsilon != None:
                    # Decay epsilon by delta
                    epsilon = max(0.05, epsilon * delta)
            for agent in agents:
                # Sample a batch from the Replay Buffer
                batch = agent.replaybuffer.sample(batch_size)
                agent.dqn.train_q_network(batch, agent.gamma)

    
    # Visualise greedy trajectory after each episode
    for i, agent in enumerate(agents):
        greedy_trajectory = agent.get_greedy_trajectory()
        agent.environment.plot_trace(greedy_trajectory)
        # Store final distance of the agent when tested with the greedy policy
        distances[i].append(agent.distance_to_goal)


# Visualise final greedy policy
for i, agent in enumerate(agents):
    greedy_trajectory = agent.get_greedy_trajectory()
    agent.environment.plot_trace(greedy_trajectory)
    
plt.figure(figsize=(10,8))
plt.plot(steps, distances[0], label=r'$R(s) = (1 - dist(s, goal)$')
plt.plot(steps, distances[1], label=r'$R(s) = (1 - dist(s, goal))^2$')
plt.plot(steps, distances[2], label='Step Reward function')
plt.xlabel('Steps'); plt.ylabel('Episode Finale Distance from Goal'); 
plt.legend()
plt.show()
