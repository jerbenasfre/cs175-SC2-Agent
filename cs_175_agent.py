import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from absl import app

from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units

from agent import Agent
from RL_brain import QLearningTable

import pickle
import time

CURRENT_AGENT_FOLDER = "cs_175_agent"
Q_TABLE_FILE_NAME = "q_table.csv"
MATCH_HISTORY_FILE_NAME = "match_history.npy"
EPISODE_COUNT_FILE_NAME = "episode_count.pickle"

class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.my_actions)
        self.load_q_table(Q_TABLE_FILE_NAME)
        
        self.new_game()

        self.load_match_history(MATCH_HISTORY_FILE_NAME)
        self.episode_count = 0
        self.load_episode_count(EPISODE_COUNT_FILE_NAME)

    def reset(self):
        super(SmartAgent, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

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
        #enemy_spawning_pool = self.get_enemy_units_by_type(obs, units.Zerg.SpawningPool)
        #enemy_roach_warren = self.get_enemy_units_by_type(obs, units.Zerg.RoachWarren)
        #enemy_hydralisk_den = self.get_enemy_units_by_type(obs, units.Zerg.HydraliskDen)
        #enemy_banelings_nest = self.get_enemy_units_by_type(obs, units.Zerg.BanelingNest)
        # enemy_completed_spawning_pool = self.get_enemy_completed_units_by_type(
        #    obs, units.Zerg.SpawningPool)
        #enemy_zerglings = self.get_enemy_units_by_type(obs, units.Zerg.Zergling)
        #enemy_banelings = self.get_enemy_units_by_type(obs, units.Zerg.Baneling)
        #enemy_hydralisks = self.get_enemy_units_by_type(obs, units.Zerg.Hydralisk)
        #enemy_roaches = self.get_enemy_units_by_type(obs, units.Zerg.Roach)
        #enemy_queens = self.get_my_units_by_type(obs, units.Zerg.Queen)
        enemy_buildings = [unit for unit in obs.observation.raw_units
                        if (unit.unit_type == units.Zerg.SpawningPool
                        or unit.unit_type == units.Zerg.Spire
                        or unit.unit_type == units.Zerg.GreaterSpire
                        or unit.unit_type == units.Zerg.BanelingNest
                        or unit.unit_type == units.Zerg.UltraliskCavern
                        or unit.unit_type == units.Zerg.HydraliskDen
                        or unit.unit_type == units.Zerg.EvolutionChamber
                        or unit.unit_type == units.Zerg.RoachWarren
                        or unit.unit_type == units.Zerg.InfestationPit
                        or unit.unit_type == units.Zerg.LurkerDen
                        or unit.unit_type == units.Zerg.SporeCrawler
                        or unit.unit_type == units.Zerg.SporeCrawlerUprooted
                        or unit.unit_type == units.Zerg.SpineCrawler
                        or unit.unit_type == units.Zerg.SpineCrawlerUprooted)
                        and unit.alliance == features.PlayerRelative.ENEMY]

        enemy_ground = [unit for unit in obs.observation.raw_units
                        if (unit.unit_type == units.Zerg.Zergling
                        or unit.unit_type == units.Zerg.Baneling
                        or unit.unit_type == units.Zerg.Hydralisk
                        or unit.unit_type == units.Zerg.Lurker
                        or unit.unit_type == units.Zerg.Roach
                        or unit.unit_type == units.Zerg.Ravager
                        or unit.unit_type == units.Zerg.Queen
                        or unit.unit_type == units.Zerg.Ultralisk
                        or unit.unit_type == units.Zerg.SwarmHost
                        or unit.unit_type == units.Zerg.BanelingCocoon
                        or unit.unit_type == units.Zerg.LurkerCocoon
                        or unit.unit_type == units.Zerg.Cocoon)
                        and unit.alliance == features.PlayerRelative.ENEMY]

        enemy_air = [unit for unit in obs.observation.raw_units
                        if (unit.unit_type == units.Zerg.Mutalisk
                        or unit.unit_type == units.Zerg.Corruptor
                        or unit.unit_type == units.Zerg.BroodLord)
                        and unit.alliance == features.PlayerRelative.ENEMY]

        return (len(command_centers),
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
                #len(enemy_spawning_pool),
                #len(enemy_hydralisk_den),
                #len(enemy_roach_warren),
                #len(enemy_banelings_nest),
                len(enemy_buildings),
                len(enemy_ground),
                len(enemy_air))

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        # Increment e_greedy after 5 episodes
        # if self.qtable.e_greedy < .90 and self.episodeCount%5 == 0:
            #self.qtable.increment_greedy()

        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        
        if obs.last():
            self.episode_count += 1
                       
            self.update_match_history(obs.reward)
            self.plot_match_history()
            
            self.save_q_table(Q_TABLE_FILE_NAME)
            self.save_match_history(MATCH_HISTORY_FILE_NAME)
            self.save_episode_count(EPISODE_COUNT_FILE_NAME)

        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward + self.custom_reward(obs),
                              'terminal' if obs.last() else state)
            
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)
    
    def custom_reward(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        
        lenarmy = len(marines) + len(marauders)
        lenwork = len(scvs)
        
        if lenwork <= 0:
            return 0
        else:
            return (lenarmy / lenwork)

    def update_match_history(self, outcome):
        """Updates lose rate, stalemate rate, and win rate lists based on the
        outcome of the current episode.
        
        self.match_history is a list of three lists. The lists contain the 
            agent's lose rates, stalement rates, and win rates per episode re-
            spectively. For example, self.match_history[0, 3] returns the
            agent's lose rate as of the 3rd episode.
            
        If self.match_history does not exist, then this function will
            initialize that variable.
            
        Otherwise, it will append the lose rate, stalement rate, and win rate
            as of the current episode to the respective lists in
            self.match_history.
        
        Arguments:
            outcome: Number in {-1.0, 0.0, 1.0} representing the outcome of the
                game. -1.0, 0.0, and 1.0 represent whether our agent lost, 
                reached a stalemate, or won, respectively.
        """
        index = int(outcome) + 1

        try:
            print("Updating match history.")
            previous_addition = self.match_history[:, self.match_history.shape[1] - 1]
            previous_addition = np.expand_dims(previous_addition, axis = 1)
            
            # Previously, we simply incremented the number of wins.
            # We figured we could calculate lose, stalemate, and win rates 
            # later by dividing the cumulative sums with the number of epi-
            # sodes. 
            # But we would be doing this calculation after every game, so it 
            # would probably be time-consuming in the long run.
            
            # previous_addition[index, 0] += 1
            # ...

            # Updating win rates incrementally would save some time, even if
            # it's by a trivial amount.
            
            # win_rate_t = 1 / t * sum_{i = 1}^t win_i
            # win_rate_t = 1 / t * (win_t + sum_{i = 1}^{t - 1} win_i)
            # win_rate_t = 1 / t * (win_t + (t - 1) win_rate_{t - 1})
            # win_rate_t = win_t / t + (t - 1) win_rate_{t - 1} / t
            # win_rate_t = win_t / t + t * win_rate_{t - 1} / t - win_rate_{t - 1} / t
            # win_rate_t = win_t / t + win_rate_{t - 1} - win_rate_{t - 1} / t
            # win_rate_t = win_rate_{t - 1} + (win_t - win_rate_{t - 1}) / t

            current_game_result = np.zeros((3, 1))
            current_game_result[index, 0] += 1
            new_addition = previous_addition + 1 / self.episode_count * \
                           (current_game_result - previous_addition) 
            
            self.match_history = np.hstack((self.match_history, new_addition))
            print("Successfully updated match history.")

        except Exception as e:
            print(f"{e}\n")
            self.match_history = np.zeros((3, 1))
            self.match_history[index, 0] += 1

    def plot_match_history(self):
        x = [i for i in range(1, self.episode_count + 1)]
        
        fig, ax = plt.subplots()

        ax.plot(x, self.match_history[0])
        ax.plot(x, self.match_history[1])
        ax.plot(x, self.match_history[2])

        ax.set_title("Match History")
        ax.legend(["Lose Rate", "Stalemate Rate", "Win Rate"])
        
        ax.xaxis.set_label_text("Number of Games Played")
        ax.yaxis.set_label_text("Percentage")

        plt.show()
            
    def get_file_path(self, file_name):
        if not os.path.exists(CURRENT_AGENT_FOLDER):
            os.makedirs(CURRENT_AGENT_FOLDER)
        destination = os.path.join(CURRENT_AGENT_FOLDER, file_name)
        return destination

    def save_q_table(self, file_name):
        # print("Writing QTable to file episode_"+str(self.episodeCount))
        # CHANGE DIRECTORY NAME
        # self.qtable.q_table.to_csv(r"C:\Users\arkse\Desktop\cs175_episodes\episode_"+str(self.episodeCount)+".csv", encoding='utf-8', index=False)
        
        destination = self.get_file_path(file_name)
        print(f"Writing Q-Table to {destination}.\n")
        self.qtable.q_table.to_csv(destination, encoding = 'utf-8', index = False)
        
    def load_q_table(self, file_name):
        try:
            target = os.path.join(CURRENT_AGENT_FOLDER, file_name)
            print(f"Attempting to load Q-Table from '{target}'.")
            self.qtable.q_table = pd.read_csv(target)
            print("Succeeded in loading Q-Table.\n")
        except Exception as e:
            print(f"{e}\n")

    def save_log(self, obs):
        
        raise NotImplementedError
        
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        refineries = self.get_my_units_by_type(obs, units.Terran.Refinery)
        tech_labs = self.get_my_units_by_type(obs, units.Terran.BarracksTechLab)

        minerals = obs.observation.player.minerals
        vespene = obs.observation.player.vespene
        
        # print("Writing state to file episode_"+str(self.episode_count)+"state.txt")
        
        file = open(r"C:\Users\arkse\Desktop\cs175_episodes\episode_"+str(self.episode_count)+"state.txt", "w")

        file.write("#CC #SCV #IdleSCV #SupplyDepots #Refineries #CompletedRefineries #CompletedSupplyDepots #Barrackses #CompletedBarrackses #Marines #Marauders QueuedMarines QueuedMarauders  FreeSupply CanAffordSuppyDepot CanAffordBarracks CanAffordMarine CanAffordMarauder CanAffordRefinery #Hatcheries #Drones #IdleDrones #Overlords #SpawningPools #HyrdaDen #RoachWarren #BanelingNest #Zergling #Banelings #Roaches #Hydralisk #Queens #Air\n")
        # file.close()
        # file = open(r"C:\Users\arkse\Desktop\cs175_episodes\episode_" + str(self.episode_count) + "state.txt", "a")
        file.write(str(self.get_state(obs)))
        file.close()

        
        #print("State:")
        #print("#CC #SCV #IdleSCV #SupplyDepots #Refineries #CompletedRefineries #CompletedSupplyDepots #Barrackses #CompletedBarrackses #Marines #Marauders QueuedMarines QueuedMarauders  FreeSupply CanAffordSuppyDepot CanAffordBarracks CanAffordMarine CanAffordMarauder CanAffordRefinery #Hatcheries #Drones #IdleDrones #Overlords #SpawningPools #HyrdaDen #RoachWarren #BanelingNest #Zergling #Banelings #Roaches #Hydralisk #Queens #Air")
        #print(self.qtable.q_table)

        print("======================================================")
        print("Marines Alive:", len(marines))
        print("Marauders Alive:", len(marauders))
        print("SCVS Alive:", len(scvs))
        print("Command Centers Up:", len(command_centers))
        print("Supply Depots Up:", len(supply_depots))
        print("Refineries Up:", len(refineries))
        print("Barrackses Up:", len(barrackses))
        print("Tech Labs Up:", len(tech_labs))
        print("Minerals:", minerals)
        print("Vespene:", vespene)
        
    def save_match_history(self, file_name):
        destination = self.get_file_path(file_name)
        print(f"Writing Match History to {destination}.\n")
        with open(destination, "wb") as f:
            np.save(f, self.match_history)
        
    def load_match_history(self, file_name):
        try:
            target = os.path.join(CURRENT_AGENT_FOLDER, file_name)
            print(f"Attempting to load Match History from '{target}'.")
            with open(target, "rb") as f:
                self.match_history = np.load(f)
            print("Succeeded in loading Match History.\n")
        except Exception as e:
            print(f"{e}\n")

    def save_episode_count(self, file_name):
        destination = self.get_file_path(file_name)
        print(f"Saving Episode Count to {destination}.\n")
        with open(destination, "wb") as f:
            pickle.dump(self.episode_count, f)
    
    def load_episode_count(self, file_name):
        try:
            target = os.path.join(CURRENT_AGENT_FOLDER, file_name)
            print(f"Attempting to load Episode Count from '{target}'.")
            with open(target, "rb") as f:
                self.episode_count = pickle.load(f)
            print("Succeeded in loading Episode Count.\n")
        except Exception as e:
            print(f"{e}\n")
    

def main(unused_argv):
    agent1 = SmartAgent()
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
                game_steps_per_episode = 38800,
                disable_fog=True,
        ) as env:
            run_loop.run_loop([agent1], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % (
            elapsed_time))

if __name__ == "__main__":
#    pd.set_option('display.max_rows', None)
#    pd.set_option('display.max_columns', None)
#    pd.set_option('display.width', None)
#    pd.set_option('display.max_colwidth', -1)
    app.run(main)
