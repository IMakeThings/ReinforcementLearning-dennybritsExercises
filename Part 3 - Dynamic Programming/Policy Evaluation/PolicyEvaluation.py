import numpy as np
import sys
# if "../" not in sys.path:
#   sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        exit = True
        # TODO: Implement!
        for state in range(env.nS):
            delta = V[state]
            v = 0
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += policy[state][action] * (reward + (discount_factor * prob * V[next_state]))
            V[state] = v

            if(abs(v - delta) > theta):
                exit = False


        if exit:
            break
    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA



print(random_policy)
v = policy_eval(random_policy, env)

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# # Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)