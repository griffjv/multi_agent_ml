from pettingzoo.magent import battle_v3
import time
from pettingzoo.utils import average_total_reward, random_demo
import random
import numpy as np


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
            y_dist = 6-min_loc[0]
            x_dist = min_loc[1]-6
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
        # return environment.action_space(agent).sample()
        return random.randint(0, 12)

def average_red_reward(environ, red_policy, blue_policy, max_episodes=100, max_steps=100):
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
        if episode%2==0:
            max_iterations = 10000
        else:
            max_iterations = 0
        for agent in environ.agent_iter(max_iter=max_iterations):
            obs, reward, done, _ = environ.last(observe=False)
            if (environ.agent_selection[0] == 'r'):
                total_reward += reward
                total_steps += 1
            if done:
                a = None
            elif isinstance(obs, dict) and 'action_mask' in obs:
                a = random.choice(np.flatnonzero(obs['action_mask']))
            else:
                if environ.agent_selection[0] == 'r':
                    a = red_policy(environ, agent)
                else:
                    a = blue_policy(environ, agent)
            environ.step(a)
            #environ.render()
            #time.sleep(.02)
        num_episodes = episode + 1
    print("Average red reward", total_reward / (num_episodes/2))

    return total_reward / (num_episodes/2)


env = battle_v3.env(map_size=24, minimap_mode=False, step_reward=-0.005,
                    dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
                    max_cycles=1000, extra_features=False)  # min size map
env.reset()
print("number of agents: ", env.num_agents)
print("agent selection: ", env.agent_selection)
print("action space: ", env.action_space(env.agent_selection))

# ACTIONS: 8 squares touching agent and 4 double jump in each cardinal direction
# ACTIONS: attack each 8 touching squares

# Test random policy and return score distribution
env.render(mode='human')

action = random_policy(env, env.agent_selection)

# avg_reward = average_total_reward(env, max_episodes=100, max_steps=10000)
env.reset()
avg_red_reward = average_red_reward(env, heuristic_policy, random_policy, max_episodes=200, max_steps=100000)

avg_red_reward = average_red_reward(env, heuristic_policy, random_policy, max_episodes=200, max_steps=100000)
