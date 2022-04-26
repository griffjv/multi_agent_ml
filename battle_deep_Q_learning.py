import dill as dill
from pettingzoo.magent import battle_v3
import matplotlib.pyplot as plt
import time
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
        self.memory.append(Transition(*args))  # save transition to replay memory

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

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

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
    obs = np.zeros((7, 7, 1))
    obs[:, :, 0] = observation[3:10, 3:10, 3]
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


def select_action(state, steps_done, policy_net):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(21)]], dtype=torch.long)


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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def dqn(env, memory, target_net, policy_net, optimizer, num_episodes=12):
    steps_done = 0
    for i_episode in range(num_episodes):
        # Create dictionary to store prev states
        prev_states = {}
        prev_actions = {}

        # Initialize the environment and state
        env.reset()
        obs = env.observe(env.agent_selection)
        parsed_obs = parse_observation(obs)
        state = observation_to_input(parsed_obs)

        if i_episode % 2 == 0:
            maximum_iters = 1000
        else:
            maximum_iters = 0
        for agent in env.agent_iter(max_iter=maximum_iters):
            if env.agent_selection[0] == 'r':
                agent_sel = env.agent_selection
                if agent_sel in prev_states:
                    # previous step
                    obs, reward, _, _ = env.last(observe=True)
                    reward = torch.tensor([reward])
                    next_state = torch.from_numpy(observation_to_input(parse_observation(obs)))
                    # Store the transition in memory
                    memory.push(prev_states[agent_sel], prev_actions[agent_sel], next_state, reward)

                    # Perform one step of the optimization (on the policy network)
                    optimize_model(memory, target_net, policy_net, optimizer)
                    steps_done += 1

                    # Select and perform an action
                    action = select_action(state, steps_done, policy_net)
                    env.step(action.numpy()[0][0])

                    # Increment
                    prev_actions[agent_sel] = action
                    prev_states[agent_sel] = next_state

                else:  # take random action for first steps
                    action = random_policy(env, agent_sel)
                    env.step(action)
                    obs, _, _, _ = env.last(observe=True)
                    env.step(action)
                    prev_actions[agent_sel] = torch.from_numpy(np.array([[action]]))
                    prev_states[agent_sel] = torch.from_numpy(observation_to_input(parse_observation(obs)))
            else:
                blue_action = random_policy(env, env.agent_selection)
                env.step(blue_action)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    return target_net, policy_net


def main():
    # setup environment
    env = battle_v3.env(map_size=12, minimap_mode=False, step_reward=-0.005,
                        dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
                        max_cycles=1000, extra_features=False)  # min size map
    env.reset()
    test_obs = env.observe(env.agent_selection)
    parsed_obs = parse_observation(test_obs)
    input_obs = observation_to_input(parsed_obs)
    n_actions = 21  # predefined

    # setup DQN model
    # TODO: Change layers of DQN to not use convolutional layers... since input is just a vec
    policy_net = DQN(input_obs.size, n_actions).to()
    target_net = DQN(input_obs.size, n_actions).to()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    target, policy = dqn(env, memory, target_net, policy_net, optimizer, num_episodes=100)

    print("Completed")


if __name__ == '__main__':
    main()
