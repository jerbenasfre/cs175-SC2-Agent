import math
import random

from collections import namedtuple

from absl import app
# from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env
import time

from cs_175_agent import Agent
import helper

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

CURRENT_AGENT_FOLDER = "dqn_agent"
MATCH_HISTORY_FILE_NAME = "match_history.npy"
EPISODE_COUNT_FILE_NAME = "episode_count.pickle"
STEP_COUNT_FILE_NAME = "step_count.pickle"
STATE_DICT_FILE_NAME = "state_dict.tar"


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.hidden_layer_size = 100

        self.fc1 = nn.Linear(state_size, self.hidden_layer_size)
        self.output = nn.Linear(self.hidden_layer_size, action_size)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = self.output(t)
        return t

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)

class RL_Agent(Agent):
    def __init__(self, strategy, device):
        super(RL_Agent, self).__init__()
        
        self.strategy = strategy
        self.num_actions = len(self.my_actions)
        self.device = device

        self.new_game()
        
        self.load_match_history(CURRENT_AGENT_FOLDER, MATCH_HISTORY_FILE_NAME)
        self.load_episode_count(CURRENT_AGENT_FOLDER, EPISODE_COUNT_FILE_NAME)
        self.load_step_count(CURRENT_AGENT_FOLDER, STEP_COUNT_FILE_NAME)
        
    def reset(self):
        super(RL_Agent, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
    
    def get_state(self, obs): # second line number - first line number + 1
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]

        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        refineries = self.get_my_units_by_type(obs, units.Terran.Refinery)
        completed_refineries = self.get_my_completed_units_by_type(obs, units.Terran.Refinery)

        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)

        queued_maruders = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100
        can_afford_marauder = obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 50
        can_afford_refinery = obs.observation.player.minerals >= 75

        enemy_drones = self.get_enemy_units_by_type(obs, units.Zerg.Drone)
        enemy_idle_drone = [drone for drone in enemy_drones if drone.order_length == 0]
        enemy_hatcheries = self.get_enemy_units_by_type(obs, units.Zerg.Hatchery)

        enemy_expanded = False

        if len(enemy_hatcheries) > 1:
            enemy_expanded = True

        enemy_hatcheries.extend(self.get_enemy_units_by_type(obs, units.Zerg.Hive))
        enemy_hatcheries.extend(self.get_enemy_units_by_type(obs, units.Zerg.Lair))
        enemy_overlords = self.get_enemy_units_by_type(
            obs, units.Zerg.Overlord)
        enemy_overlords.extend(self.get_enemy_units_by_type(obs, units.Zerg.Overseer))
        # enemy_completed_overlords = self.get_enemy_completed_units_by_type(
        #    obs, units.Zerg.Overlord)
        enemy_spawning_pool = self.get_enemy_units_by_type(obs, units.Zerg.SpawningPool)
        enemy_roach_warren = self.get_enemy_units_by_type(obs, units.Zerg.RoachWarren)
        enemy_hydralisk_den = self.get_enemy_units_by_type(obs, units.Zerg.HydraliskDen)
        enemy_banelings_nest = self.get_enemy_units_by_type(obs, units.Zerg.BanelingNest)
        # enemy_completed_spawning_pool = self.get_enemy_completed_units_by_type(
        #    obs, units.Zerg.SpawningPool)
        enemy_zerglings = self.get_enemy_units_by_type(obs, units.Zerg.Zergling)
        enemy_banelings = self.get_enemy_units_by_type(obs, units.Zerg.Baneling)
        enemy_hydralisks = self.get_enemy_units_by_type(obs, units.Zerg.Hydralisk)
        enemy_roaches = self.get_enemy_units_by_type(obs, units.Zerg.Roach)
        enemy_queens = self.get_my_units_by_type(obs, units.Zerg.Queen)

        enemy_air = self.get_enemy_units_by_type(obs, units.Zerg.Mutalisk)
        enemy_air.extend(self.get_enemy_units_by_type(obs, units.Zerg.BroodLord))
        enemy_air.extend(self.get_enemy_units_by_type(obs, units.Zerg.Corruptor))

        return torch.Tensor((len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(refineries),
                len(completed_refineries),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                len(marauders),
                queued_marines,
                queued_maruders,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                can_afford_marauder,
                can_afford_refinery,
                enemy_expanded,
                len(enemy_hatcheries),
                len(enemy_drones),
                len(enemy_idle_drone),
                len(enemy_overlords),
                # len(enemy_completed_supply_depots),
                len(enemy_spawning_pool),
                len(enemy_hydralisk_den),
                len(enemy_roach_warren),
                len(enemy_banelings_nest),
                # len(enemy_completed_barrackses),
                len(enemy_zerglings),
                len(enemy_banelings),
                len(enemy_roaches),
                len(enemy_hydralisks),
                len(enemy_queens),
                len(enemy_air))).unsqueeze(0)
    
    
    def select_action(self, obs, policy_net):
        super(RL_Agent, self).step(obs)
        state = self.get_state(obs)
        rate = self.strategy.get_exploration_rate(self.steps)

        if rate > random.random():
            print("Explore")
            # random.randrange(stop) returns a randomly selected element from
            # range(stop).
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)                # explore
            
        else:
            print("Exploit")
            with torch.no_grad():
                # return policy_net(state).argmax(dim = 1).to(self.device) # exploit
                return torch.unsqueeze(policy_net(state).argmax(), 0).to(self.device) # exploit

def extract_tensors(experiences):
    
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))
        # return policy_net(states).gather(dim = 0, index = actions.unsqueeze(0))
    
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations  = next_states.flatten(start_dim = 1) \
            .max(dim = 1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim = 1)[0].detach()
        return values

def main(unused_argv):    
    batch_size = 256
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    memory_size = 100000
    lr = 0.001
    num_episodes = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = RL_Agent(strategy, device)
    memory = ReplayMemory(memory_size)

    policy_net = DQN(34, len(agent.my_actions))
    try:
        target = os.path.join(CURRENT_AGENT_FOLDER, STATE_DICT_FILE_NAME)
        print(f"Attempting to load Policy Network from '{target}'.")
        policy_net.load_state_dict(torch.load(target))
        print("Succeeded in loading Policy Network.")
    except Exception as e:
        print(e)
    target_net = DQN(34, len(agent.my_actions))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params = policy_net.parameters(), lr = lr)
    
    start_time = time.time()
    
    try:  
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.zerg,sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=48,
                disable_fog=True,
        ) as env:
                    
            agent.setup(env.observation_spec(), env.action_spec())
            for episode in range(num_episodes):
                timesteps = env.reset()                
                agent.reset()
                state = agent.get_state(timesteps[0])
                while True:
                    action = agent.select_action(timesteps[0], policy_net)
                    print(agent.my_actions[action])
                    # Not going to be using the step function.
                    # Actually, we will probably have to use it.
                    # step_actions = [agent.step(timesteps[0])]
                    timesteps = env.step([getattr(agent, agent.my_actions[action])(timesteps[0])])
                    reward = torch.Tensor([timesteps[0].reward])
                    next_state = agent.get_state(timesteps[0])
                    
                    memory.push(Experience(state, action, next_state, reward))
                    state = next_state
                    
                    if memory.can_provide_sample(batch_size):
                        experiences = memory.sample(batch_size)
                        states, actions_, rewards, next_states = extract_tensors(experiences)
                        
                        current_q_values = QValues.get_current(policy_net, states, actions_)
                        next_q_values = QValues.get_next(target_net, next_states)
                        target_q_values = (next_q_values * gamma) + rewards
    
                        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    if timesteps[0].last():
                        agent.update_match_history(timesteps[0].reward)
                        agent.plot_match_history()

                        agent.save_match_history(CURRENT_AGENT_FOLDER, MATCH_HISTORY_FILE_NAME)
                        agent.save_episode_count(CURRENT_AGENT_FOLDER, EPISODE_COUNT_FILE_NAME)
                        agent.save_step_count(CURRENT_AGENT_FOLDER, STEP_COUNT_FILE_NAME)
                        
                        
                        destination = helper.get_file_path(CURRENT_AGENT_FOLDER, STATE_DICT_FILE_NAME)
                        torch.save(policy_net.state_dict(), destination)
                        break

                        
                if episode % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                        
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % (
            elapsed_time))
        
if __name__ == '__main__':
    app.run(main)
        
    
    
                
