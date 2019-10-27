import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        pass
        action_values = np.zeros(nA)
        action_values.fill(epsilon/(nA-1))

        best_action = np.argmax(Q[observation])

        action_values[best_action] = 1 - epsilon

        return action_values
        
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print("Episode: " + str(episode) + "/" + str(num_episodes))

        if (episode+1) % 100000 == 0:
            print("Nice")
            V = defaultdict(float)
            for state, actions in Q.items():
                action_value = np.max(actions)
                V[state] = action_value
            plotting.plot_value_function(V, title="Optimal Value Function")

        states_visited = []
        state = env.reset()

        while True:
            action_values = policy(state)
            action = np.random.choice(range(env.action_space.n), p = action_values)
            next_state, reward, done, info = env.step(action)
            
            states_visited.append([state, action, reward])

            if done:
                break
            state = next_state

        G = 0
        for i, state in enumerate(reversed(states_visited)):
            sa_pair = (state[0], action)
            G += (discount_factor*G) + state[2] 
            if states_visited[0:i-1].count(state) == 0:
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1
                Q[state[0]][action] = returns_sum[sa_pair] / returns_count[sa_pair]

                # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    return Q, policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")