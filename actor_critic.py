import random
import numpy as np

import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
import time

# from q_learning_table import QLearningTable 

from cs_175_agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

x = NotImplementedError

class ActorCriticAgent(Agent, nn.Module):
    def __init__(self, hidden_layer_size, learning_rate = 0.0002, decay = 0.98):
        Agent.__init__(self)
        nn.Module.__init__(self)

        self.gamma = decay

        self.data = []
        
        self.state_size = 28
        
        self.fc_actor_1 = nn.Linear(self.state_size, hidden_layer_size)
        self.fc_actor_2 = nn.Linear(hidden_layer_size, len(Agent.my_actions))

        self.fc_critic_1 = nn.Linear(self.state_size, hidden_layer_size)
        self.fc_critic_2 = nn.Linear(hidden_layer_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
        self.new_game()

    def reset(self):
        super(Agent, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    def v(self, x):
        x = F.relu(self.fc_critic_1(x))
        v = self.fc_critic_2(x)
        return v

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc_actor_1(x))
        x = self.fc_actor_2(x)
        prob = F.softmax(x, dim = softmax_dim)
        return prob

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100
        can_afford_refinery = obs.observation.player.minerals >= 75

        enemy_drones = self.get_enemy_units_by_type(obs, units.Zerg.Drone)
        enemy_idle_drone = [drone for drone in enemy_drones if drone.order_length == 0]
        enemy_hatcheries = self.get_enemy_units_by_type(
            obs, units.Zerg.Hatchery)
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

        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                can_afford_refinery,
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
                len(enemy_air))
    
    def step(self, obs):
        super(ActorCriticAgent, self).step(obs)
        
        state = torch.tensor(self.get_state(obs)).float()
        reward = obs.reward
        
        # Sample a' ~ pi_theta(s', a')
        prob = self.pi(state)
            
        # Example:
        # m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25]))
        # m.sample() # equal probability of 0, 1, 2, 3
        m = Categorical(prob)
        action_index = m.sample()
        action = self.my_actions[action_index]
        
        if self.previous_state != None:
            # Morvan Zhou's code does not seem to include reward?
            delta = reward + gamma * self.v(state) * (1 - obs.last()) - self.v(self.previous_state)         
            
            actor_loss = -m.log_prob(action_index) * delta
            critic_loss = delta ** 2
            
            total_loss = (actor_loss + critic_loss).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        self.previous_state = state
        self.previous_action = action
        
        return getattr(self, action)(obs)
        
def main(unused_argv):
    agent1 = ActorCriticAgent(256)
    agent2 = sc2_env.Bot(sc2_env.Race.zerg,sc2_env.Difficulty.very_easy)#RandomAgent()
    start_time = time.time()
    try:
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         agent2],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=48,
                disable_fog=True,
        ) as env:
            run_loop.run_loop([agent1], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % (
            elapsed_time))
        
if __name__ == '__main__':
    app.run(main)