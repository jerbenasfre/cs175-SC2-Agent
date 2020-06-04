import pandas as pd

import os
from absl import app

from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units

from agent import Agent
from RL_brain import QLearningTable
import helper

import time

CURRENT_AGENT_FOLDER = "cs_175_agent2b"
Q_TABLE_FILE_NAME = "q_table.csv"
MATCH_HISTORY_FILE_NAME = "match_history.npy"
EPISODE_COUNT_FILE_NAME = "episode_count.pickle"
PLOT_FILE_NAME = "figure.png"

KILL_UNIT_REWARD = 0.1
KILL_BUILDING_REWARD = 0.2
UNIT_LOST_PENALTY = -0.1
BUILDIG_LOST_PENALTY = -0.2

class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.my_actions)
        #self.qtable.epsilon = 0.4
        self.load_q_table(CURRENT_AGENT_FOLDER, Q_TABLE_FILE_NAME)

        self.new_game()

        self.load_match_history(CURRENT_AGENT_FOLDER, MATCH_HISTORY_FILE_NAME)
        self.load_episode_count(CURRENT_AGENT_FOLDER, EPISODE_COUNT_FILE_NAME)

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_unit_count = 0
        self.previous_building_count = 0

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
        # enemy_spawning_pool = self.get_enemy_units_by_type(obs, units.Zerg.SpawningPool)
        # enemy_roach_warren = self.get_enemy_units_by_type(obs, units.Zerg.RoachWarren)
        # enemy_hydralisk_den = self.get_enemy_units_by_type(obs, units.Zerg.HydraliskDen)
        # enemy_banelings_nest = self.get_enemy_units_by_type(obs, units.Zerg.BanelingNest)
        # enemy_completed_spawning_pool = self.get_enemy_completed_units_by_type(
        #    obs, units.Zerg.SpawningPool)
        # enemy_zerglings = self.get_enemy_units_by_type(obs, units.Zerg.Zergling)
        # enemy_banelings = self.get_enemy_units_by_type(obs, units.Zerg.Baneling)
        # enemy_hydralisks = self.get_enemy_units_by_type(obs, units.Zerg.Hydralisk)
        # enemy_roaches = self.get_enemy_units_by_type(obs, units.Zerg.Roach)
        # enemy_queens = self.get_my_units_by_type(obs, units.Zerg.Queen)
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
                # len(enemy_spawning_pool),
                # len(enemy_hydralisk_den),
                # len(enemy_roach_warren),
                # len(enemy_banelings_nest),
                len(enemy_buildings),
                len(enemy_ground),
                len(enemy_air))

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)

        if obs.last():
            # Increment e_greedy every 5 episodes until e_greedy == .90
            if self.qtable.epsilon < .90 and self.episodes % 5 == 0:
                self.qtable.increment_greedy()
                print("Greedy Val:", self.qtable.epsilon)

            self.update_match_history(obs.reward)
            self.plot_match_history(CURRENT_AGENT_FOLDER, PLOT_FILE_NAME)

            self.save_q_table(CURRENT_AGENT_FOLDER, Q_TABLE_FILE_NAME)
            self.save_match_history(CURRENT_AGENT_FOLDER, MATCH_HISTORY_FILE_NAME)
            self.save_episode_count(CURRENT_AGENT_FOLDER, EPISODE_COUNT_FILE_NAME)

        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward + self.custom_reward(obs),
                              'terminal' if obs.last() else state)

        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)

    def custom_reward(self, obs):
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        kill_reward = 0
        lost_penalty = 0

        unit_count = len([unit for unit in obs.observation.raw_units
                if (unit.unit_type == units.Terran.Marine or unit.unit_type == units.Terran.Marauder)
                and unit.alliance == features.PlayerRelative.SELF])

        # Not used for now. Testing unit_count first.
        #building_count = [unit for unit in obs.observation.raw_units
        #        if (unit.unit_type == units.Terran.CommandCenter
        #            or unit.unit_type == units.Terran.Barracks
        #            or unit.unit_type == units.Terran.SupplyDepot
        #            or unit.unit_type == units.Terran.Refinery)
        #        and unit.alliance == features.PlayerRelative.SELF]

        # Checking if previous action was attack so it rewards attacking.
        if killed_unit_score > self.previous_killed_unit_score and "attack_" in self.previous_action:
            kill_reward += KILL_UNIT_REWARD

        if killed_building_score > self.previous_killed_building_score and "attack_" in self.previous_action:
            kill_reward += KILL_BUILDING_REWARD

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score

        # Penalize for poor attacking or doing nothing while losing troops
        if unit_count < self.previous_unit_count and ("attack_" in self.previous_action or "do_nothing" == self.previous_action):
            lost_penalty + UNIT_LOST_PENALTY

        self.previous_unit_count = unit_count

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        #lenarmy = len(marines) + len(marauders)
        lenwork = len(scvs)

        army_reward = 0

        # Only reward army ratio when training to encourage training more. Reason for this change is because I think
        # the agent was getting points for an army existing and it could influence other actions(like do nothing).
        # This change hopefully links encourages the train action more.
        # THIS IS NOT ENOUGH. IT STILL PREFERS TO ATTACK WITH A COUPLE UNITS INSTEAD OF TRAIN MORE
        if lenwork > 0 and "train_" in self.previous_action:
            #print("Army/Worker Ratio", unit_count/lenwork)
            army_reward = (unit_count / lenwork)

        # If the agent did not train or attack, custom reward should return 0
        return army_reward+kill_reward+lost_penalty

    def save_q_table(self, folder, file_name):
        # print("Writing QTable to file episode_"+str(self.episodeCount))
        # CHANGE DIRECTORY NAME
        # self.qtable.q_table.to_csv(r"C:\Users\arkse\Desktop\cs175_episodes\episode_"+str(self.episodeCount)+".csv", encoding='utf-8', index=False)

        destination = helper.get_file_path(folder, file_name)
        #print(f"Writing Q-Table to {destination}.\n")
        self.qtable.q_table.to_csv(destination, encoding='utf-8', index=False)

    def load_q_table(self, folder, file_name):
        try:
            target = os.path.join(folder, file_name)
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

        # print("Writing state to file episode_"+str(self.episodes)+"state.txt")

        file = open(r"C:\Users\arkse\Desktop\cs175_episodes\episode_" + str(self.episodes) + "state.txt", "w")

        file.write(
            "#CC #SCV #IdleSCV #SupplyDepots #Refineries #CompletedRefineries #CompletedSupplyDepots #Barrackses #CompletedBarrackses #Marines #Marauders QueuedMarines QueuedMarauders  FreeSupply CanAffordSuppyDepot CanAffordBarracks CanAffordMarine CanAffordMarauder CanAffordRefinery #Hatcheries #Drones #IdleDrones #Overlords #SpawningPools #HyrdaDen #RoachWarren #BanelingNest #Zergling #Banelings #Roaches #Hydralisk #Queens #Air\n")
        # file.close()
        # file = open(r"C:\Users\arkse\Desktop\cs175_episodes\episode_" + str(self.episodes) + "state.txt", "a")
        file.write(str(self.get_state(obs)))
        file.close()

        # print("State:")
        # print("#CC #SCV #IdleSCV #SupplyDepots #Refineries #CompletedRefineries #CompletedSupplyDepots #Barrackses #CompletedBarrackses #Marines #Marauders QueuedMarines QueuedMarauders  FreeSupply CanAffordSuppyDepot CanAffordBarracks CanAffordMarine CanAffordMarauder CanAffordRefinery #Hatcheries #Drones #IdleDrones #Overlords #SpawningPools #HyrdaDen #RoachWarren #BanelingNest #Zergling #Banelings #Roaches #Hydralisk #Queens #Air")
        # print(self.qtable.q_table)

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


def main(unused_argv):
    agent1 = SmartAgent()
    agent2 = sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)  # RandomAgent()
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
                game_steps_per_episode=38800,
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
