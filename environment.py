import numpy as np
import cv2
import torch

BLUE = np.array((255, 0, 0))
YELLOW = np.array((0, 255, 255))
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)


# The Environment class defines the "world" within which the agent is acting
class Environment:

    # Function to initialise an Environment object
    def __init__(self, display, magnification, id=0):
        # Set whether the environment should be displayed after every step
        self.display = display
        # Set the magnification factor of the display
        self.magnification = magnification
        # Set the initial state of the agent
        self.init_state = np.array([0.15, 0.15], dtype=np.float32)
        # Set the initial state of the goal
        self.goal_state = np.array([0.75, 0.85], dtype=np.float32)

        # Set the space which the obstacle occupies
        self.obstacle_space = np.array([[0.3, 0.5], [0.3, 0.6]], dtype=np.float32)
        # Set the width and height of the environment
        self.width = 1.0
        self.height = 1.0
        self.W = int(magnification * self.width)
        self.H = int(magnification * self.height)
        self.S = int(magnification * 0.1)
        self.half_S = int(self.S / 2)

        self.pos = np.meshgrid(np.linspace(0.05, 0.95, 10).round(2), np.linspace(0.05, 0.95, 10).round(2))
        self.pos = np.stack((self.pos[0], self.pos[1]))

        self.states = torch.tensor([self.pos[:,i,j] for i in range(10) for j in range(10)]).float()
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification * self.height), int(self.magnification * self.width), 3], dtype=np.uint8)
        self.img = self.image.copy()

        self.id = id

    def plot_grid(self, img):

        for i in range(0, self.H + self.magnification, self.S):
            cv2.line(img, (i, 0), (i, self.W), WHITE, 2)
        for j in range(0, self.W + self.magnification, self.S):
            cv2.line(img, (0, j), (self.H, j), WHITE, 2)

    def state_to_pos(self, state):

        x, y = state
        x = int(x * self.magnification)
        y = self.H - int(y * self.magnification)
        return (x,y)

    def plot_Q(self, state, Q):

        x, y = self.state_to_pos(state)
        center = x, y
        # Corners of each state
        corners = [(x+self.half_S, y+self.half_S), (x-self.half_S, y+self.half_S), (x-self.half_S, y-self.half_S), (x+self.half_S, y-self.half_S)]
        # Edges corresponding to every action
        edges = {0: [corners[1], corners[0]], 1: [corners[2], corners[1]],
                 2: [corners[2], corners[3]], 3: [corners[0], corners[3]]}
        maxQ, minQ = Q.max(), Q.min()

        for action in range(4):
            col = np.array((YELLOW - BLUE) / (maxQ - minQ) * (Q[action] - maxQ) + YELLOW)
            color = tuple(col.astype('float'))

            cv2.fillConvexPoly(self.image, np.array([edges[action][0], edges[action][1], center]), color=color)
            cv2.line(self.image, edges[action][0], center, BLACK, 1)
            cv2.line(self.image, edges[action][1], center, BLACK, 1)

        self.plot_grid(self.image)

    def plot_Qs(self, Q):

        for state, Q in zip(self.states, Q):
            self.plot_Q(state, Q)

        cv2.imshow("Q function"+str(self.id), self.image)
        cv2.waitKey(1)

    def plot_trace(self, trace):
        self.img.fill(0)

        self.plot_grid(self.img)
        # Draw the obstacle
        obstacle_left = int(self.magnification * self.obstacle_space[0, 0])
        obstacle_top = int(self.magnification * (1 - self.obstacle_space[1, 1]))
        obstacle_width = int(self.magnification * (self.obstacle_space[0, 1] - self.obstacle_space[0, 0]))
        obstacle_height = int(self.magnification * (self.obstacle_space[1, 1] - self.obstacle_space[1, 0]))
        obstacle_top_left = (obstacle_left, obstacle_top)
        obstacle_bottom_right = (obstacle_left + obstacle_width, obstacle_top + obstacle_height)
        cv2.rectangle(self.img, obstacle_top_left, obstacle_bottom_right, (150, 150, 150), thickness=cv2.FILLED)
        goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.03 * self.magnification)
        cv2.circle(self.img, goal_centre, goal_radius, WHITE, cv2.FILLED)
        trace = [self.state_to_pos(s) for s in trace]
        for i in range(1, len(trace)):

            r = np.array(RED) * (len(trace) - 1 - i) / (len(trace) - 2)
            g = np.array(GREEN) * (i - 1) / (len(trace) - 2)
            col = r + g
            color = tuple(col.astype('float'))
            cv2.line(self.img, tuple(trace[i-1]), tuple(trace[i]), color, 2)

        cv2.circle(self.img, tuple(trace[0]), 10, RED, cv2.FILLED)

        cv2.circle(self.img, tuple(trace[-1]), 10, GREEN, cv2.FILLED)

        img = cv2.imshow("Trajectory"+str(self.id), self.img)
        cv2.waitKey(1)

        return img



    # Function to reset the environment, which is done at the start of each episode
    def reset(self):
        return self.init_state

    # Function to execute an agent's step within this environment, returning the next state and the distance to the goal
    def step(self, state, action):
        # Determine what the new state would be if the agent could move there
        next_state = state + action
        # If this state is outside the environment's perimeters, then the agent stays still
        if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
            next_state = state
        # If this state is inside the obstacle, then the agent stays still
        if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
            next_state = state
        # Compute the distance to the goal
        distance_to_goal = np.linalg.norm(next_state - self.goal_state)
        # Draw and show the environment, if required
        if self.display:
            self.draw(next_state)
        # Return the next state and the distance to the goal
        return next_state.round(2), distance_to_goal

    # Function to draw the environment and display it on the screen, if required
    def draw(self, agent_state):
        # Create a BLACK image
        self.image.fill(0)
        # Draw the obstacle
        obstacle_left = int(self.magnification * self.obstacle_space[0, 0])
        obstacle_top = int(self.magnification * (1 - self.obstacle_space[1, 1]))
        obstacle_width = int(self.magnification * (self.obstacle_space[0, 1] - self.obstacle_space[0, 0]))
        obstacle_height = int(self.magnification * (self.obstacle_space[1, 1] - self.obstacle_space[1, 0]))
        obstacle_top_left = (obstacle_left, obstacle_top)
        obstacle_bottom_right = (obstacle_left + obstacle_width, obstacle_top + obstacle_height)
        cv2.rectangle(self.image, obstacle_top_left, obstacle_bottom_right, (150, 150, 150), thickness=cv2.FILLED)
        # Draw the agent
        agent_centre = (int(agent_state[0] * self.magnification), int((1 - agent_state[1]) * self.magnification))
        agent_radius = int(0.02 * self.magnification)
        agent_colour = (0, 0, 255)
        cv2.circle(self.image, agent_centre, agent_radius, agent_colour, cv2.FILLED)
        # Draw the goal
        goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.02 * self.magnification)
        goal_colour = (0, 255, 0)
        cv2.circle(self.image, goal_centre, goal_radius, goal_colour, cv2.FILLED)
        # Show the image
        cv2.imshow("Environment"+str(self.id), self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)
