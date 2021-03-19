# Deep Reinforcement Learning on a Grid World

Folder ```Experiments``` contains experiments for:
* Using a Replay Buffer vs Online Learning
* Using a Target Network
* Decaying ε for ε-greedy policies for different decay factors
* Testing different types of reward functions

## Dependencies

* ```pytorch```
* ```numpy```
* ```cv2```
* ```matplotlib```

## Methods contained in the repository:
### Deep Q-Learning with Experience Replay Buffer
![DQL_ReplayBuffer](https://user-images.githubusercontent.com/71031687/111794794-427c9200-88cf-11eb-81b7-f8f622d51612.JPG)
### Deep Q-Learning with Target Network
![DQL_Qnet](https://user-images.githubusercontent.com/71031687/111794800-44465580-88cf-11eb-99b3-8796cd1bdd2a.JPG)

## Visualisations

### Visualisation of Greedy Policy
Trace of the agent from start position to goal position using its learnt greedy policy.

![dqn](https://user-images.githubusercontent.com/71031687/111641872-3d074500-8806-11eb-87f1-fbabe1900723.JPG)

### Visualisation of Q function
Each triangle in each square represents an action: N, E, W, S. Most-yellowish action is the argmax action, most-blueish action is the argmin action.

![dqn2](https://user-images.githubusercontent.com/71031687/111641878-3e387200-8806-11eb-94e9-bfba62c35aec.JPG)
