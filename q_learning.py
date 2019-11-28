from environment import MountainCar
import sys
import numpy as np
import random

np.random.seed()

def sparse_dot(x, w):
    product = 0.0
    for i, v in x.items():
        product += w[i] * v
    return product

def q_value(state,action,weights):
    return sparse_dot(state,weights[:,action])

def init_weights(mode):
    if(mode == 'tile'):
        return np.zeros((2048,3))
    return np.zeros((2,3))

def best_qvalue(state,weights,bias):
    q_values = [q_value(state,i,weights) + bias for i in range(3)]
    return q_values

def epsilon_greedy(epsilon,qvalues):
    p = random.uniform(0, 1)
    if(epsilon == 0):
        return qvalues.index(max(qvalues))
    else:
        if(p >= epsilon):
            return qvalues.index(max(qvalues))
        else:
            return random.randint(0,2)

def q_learning(mode,weights_init,episodes,max_iterations,epsilon,gamma,learning_rate,bias):
    car = MountainCar(mode = mode)
    total_reward = 0
    total_reword_list = []
    greedy_action = 0
    gradient = 0
    for e in range(episodes):
        state = car.reset()
        for m in range(max_iterations):
            greedy_action = epsilon_greedy(epsilon,best_qvalue(state,weights_init,bias))
            q_value_greedy = q_value(state,greedy_action,weights_init)+bias
            s_prime,reward,done = car.step(greedy_action)
            max_qvals = max(best_qvalue(s_prime,weights_init,bias))
            for k in state.keys():
                gradient = state[k]
                weights_init[k][greedy_action] -= gradient *(learning_rate*( q_value_greedy - (reward + gamma*max_qvals)))
            state = s_prime
            bias -= learning_rate*( q_value_greedy - (reward + gamma*max_qvals))
            total_reward += reward
            if(done):
                break
        total_reword_list.append(total_reward)
        total_reward = 0
    return weights_init,total_reword_list,bias

if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])
    bias = 0
    weights_init = init_weights(mode)
    w,r,b = q_learning(mode,weights_init,episodes,max_iterations,epsilon,gamma,learning_rate,bias)

    with open(weight_out, 'w') as weightsfile:
        weightsfile.write("%s\n" % str(b))
        for i in range(len(w)):
            for j in range(len(w[i])):
                weightsfile.write("%s\n" % str(w[i][j]))

    with open(returns_out, 'w') as returnsfile:
        for i in r:
            returnsfile.write("%s\n" % str(i))