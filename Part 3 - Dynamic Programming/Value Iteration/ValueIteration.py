import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    
    def one_step_lookahead(state, value):
        A = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[state][action]:
                A[action] += reward + (discount_factor * prob * value[next_state])

        return A


    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    
    while True:
        max_delta = 0
        for state in range(env.nS):

            A = one_step_lookahead(state, V)
            best_action = np.argmax(A)
            old_v = V[state]

            wind_sum = 0
            V[state] = max(A)
            max_delta = max(abs(V[state] - old_v), max_delta)
        if max_delta < theta:
            break
    
    for state in range(env.nS):
        A = one_step_lookahead(state, V)
        best_action = np.argmax(A)

        policy[state][best_action] = 1

    return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)