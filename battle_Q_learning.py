import dill as dill
from pettingzoo.magent import battle_v3
import matplotlib.pyplot as plt
import time
from pettingzoo.utils import average_total_reward, random_demo
import random
import numpy as np
from collections import defaultdict
import hashlib

'''
Objective: Train a policy to succeed in the battle game using Q-learning
    - Each agent has a 'state' defined by it's enemy observation and friendly observation
      use a hash function to assign unique ID to each possible state
    - Iterate through x number of episodes
        - Red actions chosen according to epsilon greedy policy, blue actions either rando or heuristic
        - Update Q-value after each observation for red 
Sources: hash function generation - Gertjan Verhoeven
'''

move_dict = {
    str([2, 0]): 0,
    str([1, -1]): 1,
    str([1, 0]): 2,
    str([1, 1]): 3,
    str([0, -2]): 4,
    str([0, -1]): 5,
    str([0, 0]): 0,
    str([0, 1]): 7,
    str([0, 2]): 8,
    str([-1, -1]): 9,
    str([-1, 0]): 10,
    str([-1, 1]): 11,
    str([-2, 0]): 12
}

attack_dict = {
    str(np.array([-1, -1])): 13,
    str(np.array([-1, 0])): 14,
    str(np.array([-1, 1])): 15,
    str(np.array([0, -1])): 16,
    str(np.array([0, 1])): 17,
    str(np.array([1, -1])): 18,
    str(np.array([1, 0])): 19,
    str(np.array([1, 1])): 20
}


def encode_state(observation):
    obs_bytes = str(observation).encode('utf-8')
    m = hashlib.md5(obs_bytes)
    state = m.hexdigest()
    return state


def update_Q(Q, agent, previous_state, previous_action,
             reward, alpha, gamma, current_state=None):
    # Q update formula:
    Q[previous_state][previous_action] += alpha * (
            reward + gamma * max(Q[current_state][:]) - Q[previous_state][previous_action])
    return Q


def parse_observation(observation):
    # which layers to save, 1, 3
    obs = np.zeros((7, 7, 1))
    obs[:, :, 0] = observation[3:10, 3:10, 3]
    # obs[:, :, 1] = observation[:, :, 3]
    return obs


def battle_q_learning(environment, num_episodes, alpha, gamma=0.99,
                      eps_start=1.0, eps_decay=0.995, eps_min=0.07):
    environment.reset()
    Q_val = {}
    # this will only train red, maybe modify to train blue at same time?
    nA = environment.action_space(environment.agent_selection).n
    # Q_val, for state action pair
    Q_val = defaultdict(lambda: np.zeros(nA))  # default dict doesn't raise key error if one doesn't exist, instead it
    # sets default value
    epsilon = eps_start

    reward_hist = []
    i_ep = 0

    # modify the below if you want to train both agents
    prev_state = -1
    prev_action = -1
    total_learning_steps = 0
    while i_ep <= num_episodes:
        enemy_obs = 0
        if i_ep % 2 == 0:
            maximum_iters = 1000
        else:
            maximum_iters = 0
        for agent in environment.agent_iter(max_iter=maximum_iters):
            if environment.agent_selection[0] == 'r':
                obs, reward, done, _ = environment.last(observe=True)

                # assign state
                parsed_obs = parse_observation(obs)
                state = encode_state(parsed_obs)
                enemy = np.sum(obs[3:10, 3:10, 3])
                if enemy > 0:
                    enemy_obs += 1
                    # update Q_val-value
                    Q_val = update_Q(Q_val, environment.agent_selection, prev_state, prev_action, reward, alpha, gamma,
                                     current_state=state)

                prev_state = state
            else:
                obs, reward, done, _ = environment.last(observe=False)

            if done:
                act = None
            elif environment.agent_selection[0] == 'r':
                act = epsilon_greedy_policy(environment, agent, enemy, Q_val, state, epsilon)
                prev_action = act
            else:
                act = random_policy(environment, environment.agent_selection)
            environment.step(act)
        environment.reset()
        prev_state = -1
        prev_action = -1
        total_learning_steps += enemy_obs
        if i_ep % 50 == 0:
            r = average_red_reward(environment, q_policy, random_policy, Q_val, max_episodes=10, max_steps=100000)
            reward_hist.append(r)
            print("episode number: ", i_ep)
            print("size of Q_val: ", len(Q_val))
            print("Enemy observations: ", enemy_obs)
        i_ep += 1
        epsilon = max(1 - i_ep / num_episodes, eps_min)
    return Q_val, reward_hist, total_learning_steps


def q_policy(environment, agent, Q_value, state):
    parsed_obs = parse_observation(environment.observe(agent))
    enemy_count = np.sum(parsed_obs)
    if enemy_count > 0:
        action_qs = Q_value[state][:]  # Q values of all actions available
        max_val = max(action_qs)
        max_indices = np.where(action_qs == max_val)
        act = random.choice(max_indices[0])
    else:
        act = random.randint(0, 12)
    return act


def epsilon_greedy_policy(environment, agent, enemy_count, Q_val, state, eps):
    if random.random() < eps:
        # take random action
        act = environment.action_space(agent).sample()
        if enemy_count == 0:
            # take heuristic policy
            act = heuristic_policy(environment, agent)
    else:
        action_qs = Q_val[state][:]  # Q values of all actions available
        max_val = max(action_qs)
        max_indices = np.where(action_qs == max_val)
        act = random.choice(max_indices[0])
    return act


def random_policy(environment, agent):
    return environment.action_space(agent).sample()


def heuristic_policy(environment, agent):
    '''
    First, see if agent observation, 3rd layer(obs[:,:,3]) contains any 1's
    1's signify enemy in territory

    if 1's in obs[:,:,3]:
        if 1's within attack range:
            attack weakest enemy in range
        else:
            move towards the nearest enemy
    else:
        return random movement action
    '''
    attack_threshold = 2
    observation = env.observe(agent)
    if np.sum(observation[:, :, 3]) > 0:
        indices = np.argwhere(observation[:, :, 3])
        min_distance = 100
        min_loc = np.array([6, 6])
        for index in indices:
            dist = np.linalg.norm(index - np.array([6, 6]))
            if dist < min_distance:
                min_distance = dist
                min_loc = index
        if min_distance < attack_threshold:
            # attack
            return attack_dict[str(min_loc - [6, 6])]
        else:
            # move to closest
            y_dist = 6 - min_loc[0]
            x_dist = min_loc[1] - 6
            if abs(x_dist) > abs(y_dist):
                if abs(x_dist) > 2:
                    act = move_dict[str([0, np.sign(x_dist) * 2])]
                else:
                    act = move_dict[str([0, x_dist])]
            else:
                if abs(y_dist) > 2:
                    act = move_dict[str([np.sign(y_dist) * 2, 0])]
                else:
                    act = move_dict[str([y_dist, 0])]
            return act
    else:
        return random.randint(0, 12)


def average_red_reward(environ, red_policy, blue_policy, Q, visualize=False, max_episodes=100, max_steps=100):
    '''
    Runs an env object with random actions until either max_episodes or
    max_steps is reached. Calculates the average total reward over the
    episodes, only for the red team for the purpose of evaluating the battle
    environment
    '''
    total_reward = 0
    total_steps = 0
    done = False

    for episode in range(max_episodes):
        if total_steps >= max_steps:
            break

        environ.reset()
        # run for red but not blue
        if episode % 2 == 0:
            max_iterations = 1000
        else:
            max_iterations = 1
        for agent in environ.agent_iter(max_iter=max_iterations):
            obs, reward, done, _ = environ.last(observe=False)
            if (environ.agent_selection[0] == 'r' and episode % 2 == 0):
                total_reward += reward
                total_steps += 1
            if done:
                a = None
            elif isinstance(obs, dict) and 'action_mask' in obs:
                a = random.choice(np.flatnonzero(obs['action_mask']))
            else:
                if environ.agent_selection[0] == 'r':
                    obs = environ.observe(agent)
                    parsed_obs = parse_observation(obs)
                    state = encode_state(parsed_obs)
                    a = red_policy(environ, agent, Q, state)
                else:
                    a = blue_policy(environ, agent)
            environ.step(a)
            if visualize:
                environ.render()
                time.sleep(.02)

        num_episodes = episode + 1
    print("Average red reward", total_reward / (num_episodes / 2))

    return total_reward / (num_episodes / 2)


random.seed(123)
env = battle_v3.env(map_size=12, minimap_mode=False, step_reward=-0.005,
                    dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
                    max_cycles=1000, extra_features=False)  # min size map
full_run = 1
env.reset()
iterats = 10000
# train q
if full_run == 1:
    Q, rewards_history, total_steps = battle_q_learning(env, iterats, 0.1)
    print("length of Q: ", len(Q))
    print("total steps: ", total_steps)
    with open('Q.pkl', 'wb') as f:
        dill.dump(Q, f)
else:
    with open('Q.pkl', 'rb') as f:
        Q = dill.load(f)
# evaluate q policy
# avg_red_reward = average_red_reward(env, q_policy, random_policy, Q, max_episodes=10, max_steps=100000)
# avg_red_reward = average_red_reward(env, q_policy, random_policy, Q, visualize=True, max_episodes=2, max_steps=100000)


x_points = np.linspace(0, iterats + 50, int((iterats + 50) / 50))
plt.plot(x_points, rewards_history)
plt.xlabel('Q-Learning Episodes')
plt.ylabel('Average Reward')
plt.title('Q-learning')
plt.show()
# TODO: Ideas - 1) learn off heuristic policy to reach rewards faster. 2) do a 1 layer observation...
# shrink observation range
# IDEA: when nnoo emmmmmmmmmmmmmmmmmmmmmmmmmmmmy  observations are made, make random movement action


#### NOTES #####
##withh 1 obserrvation laayeer, number of states stabilizes at 6000,, 300 iteers
## take actions to get into enemy range... then apply Q-learrnning(ie state with 0 observations has no optimal aactioonnnnn cause no innfo)
