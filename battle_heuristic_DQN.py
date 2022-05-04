import dill as dill
from pettingzoo.magent import battle_v3
import matplotlib.pyplot as plt
import time
import statistics
from pettingzoo.utils import average_total_reward, random_demo
import random
import numpy as np
from collections import defaultdict, deque, namedtuple
import hashlib
import math
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
Objective: Train a policy to succeed in the battle game using Q-learning
    - Each agent has a 'state' defined by it's enemy observation and friendly observation
      use a hash function to assign unique ID to each possible state
    - Iterate through x number of episodes
        - Red actions chosen according to epsilon greedy policy, blue actions either rando or heuristic
        - Update Q-value after each observation for red 
Sources: hash function generation - Gertjan Verhoeven
'''


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, obs_size, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.head = nn.Linear(obs_size, outputs)

        layer_sizes = [obs_size, 64, outputs]
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        self.network = nn.Sequential(*layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).


#   def forward(self, x):
#       x = x.to()
#       return self.head(self.network(x))


def build_nn(obs_size, outputs):
    layer_sizes = [obs_size, 64, outputs]
    layers = []
    for index in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
        act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
        layers += (linear, act)
    return nn.Sequential(*layers)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 20

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


def parse_observation(observation):
    # which layers to save, 1, 3
    obs = np.zeros((13, 13, 1))
    obs[:, :, 0] = observation[:, :, 3]
    # obs[:, :, 1] = observation[:, :, 3]
    return obs


def observation_to_input(observation):
    return np.resize(observation, (1, observation.size))[0]


def encode_state(observation):
    obs_bytes = str(observation).encode('utf-8')
    m = hashlib.md5(obs_bytes)
    state = m.hexdigest()
    return state


def random_policy(environment, agent):
    return environment.action_space(agent).sample()


def select_action(state, steps_done, policy_net, eps_thresh):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_thresh:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            Qp = policy_net(state.float())
            Q, A = torch.max(Qp, axis=0)
            return A
    else:
        return torch.tensor(random.randrange(21), dtype=torch.int)


def optimize_model(memory, target_net, policy_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_values = torch.empty((BATCH_SIZE), dtype=float)
    next_state_values = torch.empty((BATCH_SIZE), dtype=float)
    for i in range(BATCH_SIZE):
        qp = policy_net(batch.state[i].float())
        pred_return, _ = torch.max(qp, axis=0)

        qt = target_net(batch.next_state[i].float())
        pred_return2, _ = torch.max(qt, axis=0)
        n_value = (pred_return2 * GAMMA) + batch.reward[i]

        state_values[i] = pred_return
        next_state_values[i] = n_value
    # next_state_values = torch.zeros(BATCH_SIZE)
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


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
    observation = environment.observe(agent)
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


def evaluate1(env, policy_net, num_episodes=200):
    steps_done = 0
    reward_history = []
    iters_history = []
    blues_defeated = 0
    eps_thresh = 0.01
    for i_episode in range(num_episodes):
        print("Episode Number: ", i_episode)
        # Create dictionary to store prev states
        prev_states = {}
        prev_actions = {}

        # Initialize the environment and state
        env.reset()
        obs = env.observe(env.agent_selection)
        parsed_obs = parse_observation(obs)
        state = observation_to_input(parsed_obs)

        if i_episode % 2 == 0:
            maximum_iters = 1600
        else:
            maximum_iters = 0
        total_reward = 0
        iters = 0
        for agent in env.agent_iter(max_iter=maximum_iters):
            if env.agent_selection[0] == 'b':
                agent_sel = env.agent_selection
                if agent_sel in prev_states:
                    # previous step
                    obs, reward, done, _ = env.last(observe=True)
                    total_reward += reward
                    reward = torch.tensor([reward])
                    next_state = torch.from_numpy(observation_to_input(parse_observation(obs)))
                    # Store the transition in memory

                    # Perform one step of the optimization (on the policy network)
                    steps_done += 1

                    # Select and perform an action
                    if done:
                        # action = None
                        env.step(None)
                    else:
                        action = select_action(next_state, steps_done, policy_net, eps_thresh)
                        # print("Action: ", action.item())
                        env.step(action.item())

                    # Increment
                    prev_actions[agent_sel] = action
                    prev_states[agent_sel] = next_state

                else:  # take random action for first steps
                    action = random_policy(env, agent_sel)
                    env.step(action)
                    obs, _, _, _ = env.last(observe=True)
                    env.step(action)
                    prev_actions[agent_sel] = torch.tensor(action)
                    prev_states[agent_sel] = torch.from_numpy(observation_to_input(parse_observation(obs)))
            else:
                _, _, done, _ = env.last(observe=False)
                if done:
                    blue_action = None
                    blues_defeated += 1
                else:
                    blue_action = heuristic_policy(env, env.agent_selection)
                env.step(blue_action)
            iters+=1
            if iters % 30 == 0:

                env.render()
                time.sleep(.5)
                env.close()
        if i_episode % 2 == 0:
            print("Reward: ", total_reward)
            print("Blue Defeated: ", blues_defeated)


            reward_history.append(total_reward)
            print("Iterations Survived: ", iters)
            iters_history.append(iters)
    print("Analysis Complete!")
    return reward_history, iters_history

def evaluate2(env, policy_net, num_episodes=200):
    steps_done = 0
    reward_history = []
    iters_history = []
    blues_defeated = 0
    eps_thresh = 0.01
    for i_episode in range(num_episodes):
        print("Episode Number: ", i_episode)
        # Create dictionary to store prev states
        prev_states = {}
        prev_actions = {}

        # Initialize the environment and state
        env.reset()
        obs = env.observe(env.agent_selection)
        parsed_obs = parse_observation(obs)
        state = observation_to_input(parsed_obs)

        if i_episode % 2 == 0:
            maximum_iters = 1600
        else:
            maximum_iters = 0
        total_reward = 0
        iters = 0
        for agent in env.agent_iter(max_iter=maximum_iters):
            if env.agent_selection[0] == 'b':
                agent_sel = env.agent_selection
                if agent_sel in prev_states:
                    # previous step
                    obs, reward, done, _ = env.last(observe=True)
                    total_reward += reward
                    reward = torch.tensor([reward])
                    next_state = torch.from_numpy(observation_to_input(parse_observation(obs)))
                    # Store the transition in memory

                    # Perform one step of the optimization (on the policy network)
                    steps_done += 1

                    # Select and perform an action
                    if done:
                        # action = None
                        env.step(None)
                    else:
                        action = select_action(next_state, steps_done, policy_net, eps_thresh)
                        # print("Action: ", action.item())
                        env.step(action.item())

                    # Increment
                    prev_actions[agent_sel] = action
                    prev_states[agent_sel] = next_state

                else:  # take random action for first steps
                    action = random_policy(env, agent_sel)
                    env.step(action)
                    obs, _, _, _ = env.last(observe=True)
                    env.step(action)
                    prev_actions[agent_sel] = torch.tensor(action)
                    prev_states[agent_sel] = torch.from_numpy(observation_to_input(parse_observation(obs)))
            else:
                _, _, done, _ = env.last(observe=False)
                if done:
                    blue_action = None
                    blues_defeated += 1
                else:
                    blue_action = random_policy(env, env.agent_selection)
                env.step(blue_action)
            iters+=1
        if i_episode % 2 == 0:
            print("Reward: ", total_reward)
            print("Blue Defeated: ", blues_defeated)
            env.close()
            env.render()
            reward_history.append(total_reward)
            print("Iterations Survived: ", iters)
            iters_history.append(iters)
    print("Analysis Complete!")
    return reward_history, iters_history



def dqn(env, memory, target_net, policy_net, optimizer, num_episodes=12):
    steps_done = 0
    loss = 0
    loss_history = []
    reward_history = []
    iters_history = []
    blues_defeated = 0
    eps_thresh = 0.99
    for i_episode in range(num_episodes):
        print("Episode Number: ", i_episode)
        # Create dictionary to store prev states
        prev_states = {}
        prev_actions = {}

        # Initialize the environment and state
        env.reset()
        obs = env.observe(env.agent_selection)
        parsed_obs = parse_observation(obs)
        state = observation_to_input(parsed_obs)

        if i_episode % 2 == 0:
            maximum_iters = 1600
        else:
            maximum_iters = 0
        total_reward = 0
        iters = 0
        for agent in env.agent_iter(max_iter=maximum_iters):
            #env.render()
            if env.agent_selection[0] == 'b':
                agent_sel = env.agent_selection
                if agent_sel in prev_states:
                    # previous step
                    obs, reward, done, _ = env.last(observe=True)
                    total_reward += reward
                    reward = torch.tensor([reward])
                    next_state = torch.from_numpy(observation_to_input(parse_observation(obs)))
                    # Store the transition in memory
                    memory.push(prev_states[agent_sel], prev_actions[agent_sel], next_state, reward)

                    # Perform one step of the optimization (on the policy network)
                    loss = optimize_model(memory, target_net, policy_net, optimizer)
                    steps_done += 1

                    # Select and perform an action
                    if done:
                        # action = None
                        env.step(None)
                    else:
                        action = select_action(next_state, steps_done, policy_net, eps_thresh)
                        # print("Action: ", action.item())
                        env.step(action.item())

                    # Increment
                    prev_actions[agent_sel] = action
                    prev_states[agent_sel] = next_state

                else:  # take random action for first steps
                    action = random_policy(env, agent_sel)
                    env.step(action)
                    obs, _, _, _ = env.last(observe=True)
                    env.step(action)
                    prev_actions[agent_sel] = torch.tensor(action)
                    prev_states[agent_sel] = torch.from_numpy(observation_to_input(parse_observation(obs)))
            else:
                _, _, done, _ = env.last(observe=False)
                if done:
                    blue_action = None
                    blues_defeated += 1
                else:
                    blue_action = heuristic_policy(env, env.agent_selection)
                env.step(blue_action)
            iters+=1
        if i_episode % 2 == 0:
            print("Reward: ", total_reward)
            print("Loss: ", loss)
            print("Blue Defeated: ", blues_defeated)
            #env.close()
            #env.render()
            loss_history.append(loss)
            reward_history.append(total_reward)
            eps_thresh -= 2 / num_episodes
            eps_thresh = max(0.05, eps_thresh)
            print("Iterations Survived: ", iters)
            iters_history.append(iters)
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    print("DQN Complete!")
    return target_net, policy_net, loss_history, reward_history, iters_history


def main():
    # setup environment
    env = battle_v3.env(map_size=20, minimap_mode=False, step_reward=-0.005,
                        dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
                        max_cycles=1000, extra_features=False)  # min size map
    env.reset()
    test_obs = env.observe(env.agent_selection)
    parsed_obs = parse_observation(test_obs)
    input_obs = observation_to_input(parsed_obs)
    n_actions = 21  # predefined
    env.render()
    env.reset()
    env.render()
    time.sleep(2)
    full_run = 0
    # setup DQN model
    # policy_net = DQN(input_obs.size, n_actions).to()
    # target_net = DQN(input_obs.size, n_actions).to()
    if full_run == 1:
        policy_net = build_nn(input_obs.size, n_actions)
        target_net = build_nn(input_obs.size, n_actions)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters())
        memory = ReplayMemory(10000)
        iterats = 1000
        target, policy, loss_history, reward_history, iters_hist = dqn(env, memory, target_net, policy_net, optimizer,
                                                           num_episodes=iterats)
    if full_run == 0:
        with open('policy_net.pkl', 'rb') as f:
            policy_net = dill.load(f)

    reward_eval, survival_eval = evaluate1(env, policy_net, num_episodes=10)

    mean_reward = statistics.mean(reward_eval)
    reward_stand_dev = statistics.stdev(reward_eval)
    mean_survival = statistics.mean(survival_eval)
    survival_stand_dev = statistics.stdev(survival_eval)
    print("DQN vs Heuristic: ")
    print("DQN Policy Reward: ", mean_reward, "+-", reward_stand_dev)
    print("DQN Policy Survival: ", mean_survival, "+-", survival_stand_dev)

    reward_eval, survival_eval = evaluate2(env, policy_net, num_episodes=200)

    mean_reward = statistics.mean(reward_eval)
    reward_stand_dev = statistics.stdev(reward_eval)
    mean_survival = statistics.mean(survival_eval)
    survival_stand_dev = statistics.stdev(survival_eval)

    print("DQN vs Random: ")
    print("DQN Policy Reward: ", mean_reward, "+-", reward_stand_dev)
    print("DQN Policy Survival: ", mean_survival, "+-", survival_stand_dev)

    x_points = np.linspace(0, (iterats / 2), int(iterats / 2 ))
    fig1 = plt.figure()
    plt.plot(x_points, reward_history)
    plt.xlabel('DQN Episodes')
    plt.ylabel('Reward')
    plt.title('Deep Q-Network Learning')
    plt.savefig('Reward_DQN.png')
    plt.show(block=False)
    time.sleep(1)
    plt.close(fig1)
    fig1 = plt.figure()
    plt.plot(x_points, iters_hist)
    plt.xlabel('DQN Episodes')
    plt.ylabel('Iterations Survived')
    plt.title('Deep Q-Network Learning')
    plt.savefig('Survival_DQN.png')
    plt.show(block=False)
    time.sleep(1)
    plt.close(fig1)
    fig1 = plt.figure()
    plt.plot(x_points, loss_history)
    plt.xlabel('DQN Episodes')
    plt.ylabel('Policy Loss')
    plt.title('Deep Q-Network Learning')
    plt.show(block=False)

    time.sleep(1)
    plt.close(fig1)
    with open('policy_net2.pkl', 'wb') as f:
        dill.dump(policy_net, f)
    print("Completed")

if __name__ == '__main__':
    main()
