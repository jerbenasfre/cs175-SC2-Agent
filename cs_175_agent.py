import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
import time

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.75):# Change from .9 to .75
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]# Get current state and associated action: reward values for that state
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)# Choose the highest rewarding action for this state(randomly select if ties)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]# Get previous state and previous action
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()# Get current state and action with maximum value?
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)# Assign new reward to action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))

class Agent(base_agent.BaseAgent):
    my_actions = ("do_nothing",
                  "harvest_minerals",
                  "harvest_vespene",
                  "build_refinery",
                  "build_supply_depot",
                  "build_barracks",
                  "build_tech_lab",
                  # =============================================================
                  #"build_expansion",
                  #"build_engineering_bay",
                  # =============================================================
                  "train_marine",
                  "train_scv",
                  "train_marauder",
                  "attack",
                  #=============================================================
                  # "attack_expansion1",
                  # "attack_expansion2",
                  #=============================================================
                  "attack_all",
                  #=============================================================
                  # "attack_all_expansion1",
                  # "attack_all_expansion2",
                  #=============================================================
                  "attack_marine",
                  #=============================================================
                  # "attack_marine_expansion1",
                  # "attack_marine_expansion2",
                  #=============================================================
                  "attack_marine_all",
                  #=============================================================
                  # "attack_marine_all_expansion1",
                  # "attack_marine_all_expansion2",
                  #=============================================================
                  "attack_marauder",
                  #=============================================================
                  # "attack_marauder_expansion1",
                  # "attack_marauder_expansion2",
                  #=============================================================
                  "attack_marauder_all")
                  # "attack_marauder_all_expansion1",
                  # "attack_marauder_all_expansion2")
    episodeCount = 1

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_enemy_units(self, obs):
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]

            scv = random.choice(idle_scvs)
            distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # How do we track if there are 3 scv gathering vespene? Is there a way to check vespene gather rate? Way to check if workers assigned to vespene?
    def harvest_vespene(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        refineries  = self.get_my_units_by_type(obs, units.Terran.Refinery)
        if len(idle_scvs) > 0 and len(refineries) != 0:
            refinery = refineries[0]
            scv = random.choice(idle_scvs)
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, refinery.tag)
        return actions.RAW_FUNCTIONS.no_op()

# BUILD ACTIONS #######################################################################################################################

    def build_expansion(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)

        if len(command_centers) < 2 and obs.observation.player.minerals >= 150 and len(scvs) > 0:

            if self.base_top_left:
                locations = [(44,22),(44,21),(44,20),(44,19),(45,18),(45,19),(45,20),(45,21)]
            else:
                locations = [(14,47),(15,47),(15,48),(16,47),(16,48),(18,50)]#(17,48)

            expansion_location = random.choice(locations)
            print("EXPANSION LOCATION:",expansion_location)
            distances = self.get_distances(obs, scvs, expansion_location)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
                "now", scv.tag, expansion_location)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        # Always build at least 4 supply depots at designated position
        if (len(supply_depots) < 4 and obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            locations_top_left = [(22, 26), (22, 28), (20, 26), (20, 28)]  # (22, 21)
            locations_bottom_right = [(35, 42), (35, 44), (33, 42), (33, 44)]
            if self.base_top_left:
                location = random.choice(locations_top_left)
            else:
                location = random.choice(locations_bottom_right)
            #print("LOCATION CHOSEN:", location)
            supply_depot_xy = location
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)

        # If supply is capped, build suppy depot at random position
        elif free_supply == 0 and len(scvs) > 0 and obs.observation.player.minerals >=100:
            supply_depot_xy = (random.randint(0, 83), random.randint(0, 83))
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)


        return actions.RAW_FUNCTIONS.no_op()

    def build_refinery(self, obs):
        refinery = self.get_my_units_by_type(obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(refinery) == 0 and obs.observation.player.minerals >= 75 and
                len(scvs) > 0):
            geysers = [unit for unit in obs.observation.raw_units
                       if unit.unit_type in [
                           units.Neutral.ProtossVespeneGeyser,
                           units.Neutral.PurifierVespeneGeyser,
                           units.Neutral.RichVespeneGeyser,
                           units.Neutral.ShakurasVespeneGeyser,
                           units.Neutral.SpacePlatformGeyser,
                           units.Neutral.VespeneGeyser
                       ]]
            scv = random.choice(scvs)
            distances = self.get_distances(obs, geysers, (scv.x, scv.y))
            geyser = geysers[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Refinery_pt(
                "now", scv.tag, geyser.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) < 2 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):

            if self.base_top_left:
                locations = [(22, 21),(25,21)]
            else:
                locations = [(35, 47), (32, 47)]
            barracks_xy = random.choice(locations)

            #print("BARRACK LOCATION CHOSEN :", barracks_xy)

            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_tech_lab(self, obs):
        completed_barracks = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)

        tech_labs = self.get_my_units_by_type(obs, units.Terran.BarracksTechLab)
        #scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_barracks) > 0 and len(tech_labs) < 2 and
                obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 50):

            if self.base_top_left:
                locations = [(22, 21), (25, 21)]
            else:
                locations = [(35, 47), (32, 47)]
            barracks = random.choice(self.get_my_units_by_type(obs, units.Terran.Barracks))

            return actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick(
                "now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

# TRAIN ACTIONS #######################################################################################################################

    def train_scv(self, obs):
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        command_centers.extend(self.get_my_units_by_type(obs, units.Terran.OrbitalCommand))
        scv_count = len(self.get_my_units_by_type(obs, units.Terran.SCV))
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(command_centers) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0 and scv_count < 19) :
            command_center = random.choice(command_centers)
            if command_center.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)

            if len(barrackses) == 2:
                if barrackses[0].order_length < 5 or barrackses[1].order_length < 5:
                    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", [barracks.tag for barracks in barrackses])
            elif barrackses[0].order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrackses[0].tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marauder(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and obs.observation.player.vespene >= 25
                and free_supply > 2):
            barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)

            # Make a check for techlab
            #barrackses

            if len(barrackses) == 2:
                if barrackses[0].order_length < 5 or barrackses[1].order_length < 5:
                    return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", [barracks.tag for barracks in barrackses])
            elif barrackses[0].order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barrackses[0].tag)
        return actions.RAW_FUNCTIONS.no_op()

# ATTACK ACTIONS #######################################################################################################################

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        if len(marines) > 0 or len(marauders) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                units = marines + marauders
                center = random.choice(units)
            else:
                center = comcenter[0]
            enemy_units = self.get_enemy_units(obs)
            distances = self.get_distances(obs, enemy_units, (center.x, center.y))
            enemy_target = enemy_units[np.argmin(distances)]
            army = []

            if len(marines) > 0:
                army.extend(marines)
            if len(marauders) > 0:
                army.extend(marauders)
            distances = self.get_distances(obs, army, (enemy_target.x, enemy_target.y))
            unit = army[np.argmax(distances)]
            #enemy_location = self.get_enemy_units(obs)

            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", unit.tag, (enemy_target.x + x_offset, enemy_target.y + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

#===============================================================================
#     def attack_expansion1(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marines) > 0 or len(marauders) > 0:
#             attack_xy = (18,44) if self.base_top_left else (40,23)
#             army = []
# 
#             if len(marines) > 0:
#                 army.extend(marines)
#             if len(marauders) > 0:
#                 army.extend(marauders)
#             distances = self.get_distances(obs, army, attack_xy)
#             unit = army[np.argmax(distances)]
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", unit.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
# 
#     def attack_expansion2(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marines) > 0 or len(marauders) > 0:
#             attack_xy = (40, 23) if self.base_top_left else (18,44)
#             army = []
# 
#             if len(marines) > 0:
#                 army.extend(marines)
#             if len(marauders) > 0:
#                 army.extend(marauders)
#             distances = self.get_distances(obs, army, attack_xy)
#             unit = army[np.argmax(distances)]
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", unit.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
#===============================================================================

    def attack_all(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        if len(marines) > 0 or len(marauders) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                units = marines + marauders
                center = random.choice(units)
            else:
                center = comcenter[0]
            enemy_units = self.get_enemy_units(obs)
            distances = self.get_distances(obs, enemy_units, (center.x, center.y))
            enemy_target = enemy_units[np.argmin(distances)]
            # enemy_location = self.get_enemy_units(obs)
            army = []

            if len(marines) > 0:
                army.extend(marines)
            if len(marauders) > 0:
                army.extend(marauders)

            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            # Only get 10 marines, randomly selected
            army = [unit.tag for unit in army]
            random.shuffle(army)
            army = army[:11]
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", army, (enemy_target.x + x_offset, enemy_target.y + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

#===============================================================================
#     def attack_all_expansion1(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marines) > 0:
#             attack_xy = (18, 44) if self.base_top_left else (40, 23)
#             army = []
# 
#             if len(marines) > 0:
#                 army.extend(marines)
#             if len(marauders) > 0:
#                 army.extend(marauders)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             # Only get 10 marines, randomly selected
#             army = [unit.tag for unit in army]
#             random.shuffle(army)
#             army = army[:11]
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", army, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
# 
#     def attack_all_expansion2(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marines) > 0:
#             attack_xy = (40, 23) if self.base_top_left else (18, 44)
#             army = []
# 
#             if len(marines) > 0:
#                 army.extend(marines)
#             if len(marauders) > 0:
#                 army.extend(marauders)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             # Only get 10 marines, randomly selected
#             army = [unit.tag for unit in army]
#             random.shuffle(army)
#             army = army[:11]
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", army, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
#===============================================================================

    def attack_marine(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                units = marines + marauders
                center = random.choice(units)
            else:
                center = comcenter[0]
            enemy_units = self.get_enemy_units(obs)
            distances = self.get_distances(obs, enemy_units, (center.x, center.y))
            enemy_target = enemy_units[np.argmin(distances)]
            
            distances = self.get_distances(obs, marines, (enemy_target.x, enemy_target.y))
            marine = marines[np.argmax(distances)]

            #enemy_location = self.get_enemy_units(obs)

            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (enemy_target.x + x_offset, enemy_target.y + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

#===============================================================================
#     def attack_marine_expansion1(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         if len(marines) > 0:
#             attack_xy = (18,44) if self.base_top_left else (40,23)
#             distances = self.get_distances(obs, marines, attack_xy)
#             marine = marines[np.argmax(distances)]
# 
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
# 
#     def attack_marine_expansion2(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         if len(marines) > 0:
#             attack_xy = (40, 23) if self.base_top_left else (18,44)
#             distances = self.get_distances(obs, marines, attack_xy)
#             marine = marines[np.argmax(distances)]
# 
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
#===============================================================================

    def attack_marine_all(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                units = marines + marauders
                center = random.choice(units)
            else:
                center = comcenter[0]
            enemy_units = self.get_enemy_units(obs)
            distances = self.get_distances(obs, enemy_units, (center.x, center.y))
            enemy_target = enemy_units[np.argmin(distances)]
            
            distances = self.get_distances(obs, marines, (enemy_target.x, enemy_target.y))
            # enemy_location = self.get_enemy_units(obs)

            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            # Only get 10 marines, randomly selected
            marines = [marine.tag for marine in marines]
            random.shuffle(marines)
            marines = marines[:11]
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marines, (enemy_target.x + x_offset, enemy_target.y + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

#===============================================================================
#     def attack_marine_all_expansion1(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         if len(marines) > 0:
#             attack_xy = (18, 44) if self.base_top_left else (40, 23)
#             distances = self.get_distances(obs, marines, attack_xy)
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             # Only get 10 marines, randomly selected
#             marines = [marine.tag for marine in marines]
#             random.shuffle(marines)
#             marines = marines[:11]
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marines, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
# 
#     def attack_marine_all_expansion2(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         if len(marines) > 0:
#             attack_xy = (40, 23) if self.base_top_left else (18, 44)
#             distances = self.get_distances(obs, marines, attack_xy)
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             # Only get 10 marines, randomly selected
#             marines = [marine.tag for marine in marines]
#             random.shuffle(marines)
#             marines = marines[:11]
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marines, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
#===============================================================================

    def attack_marauder(self, obs):
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        if len(marauders) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                units = marines + marauders
                center = random.choice(units)
            else:
                center = comcenter[0]
            enemy_units = self.get_enemy_units(obs)
            distances = self.get_distances(obs, enemy_units, (center.x, center.y))
            enemy_target = enemy_units[np.argmin(distances)]
            
            distances = self.get_distances(obs, marauders, (enemy_target.x, enemy_target.y))
            marauder = marauders[np.argmax(distances)]

            # enemy_location = self.get_enemy_units(obs)

            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marauder.tag, (enemy_target.x + x_offset, enemy_target.y + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

#===============================================================================
#     def attack_marauder_expansion1(self, obs):
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marauders) > 0:
#             attack_xy = (18,44) if self.base_top_left else (40,23)
#             distances = self.get_distances(obs, marauders, attack_xy)
#             marauder = marauders[np.argmax(distances)]
# 
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marauder.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
# 
#     def attack_marauder_expansion2(self, obs):
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marauders) > 0:
#             attack_xy = (40, 23) if self.base_top_left else (18,44)
#             distances = self.get_distances(obs, marauders, attack_xy)
#             marauder = marauders[np.argmax(distances)]
# 
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marauder.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
#===============================================================================

    def attack_marauder_all(self, obs):
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        if len(marauders) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                units = marines + marauders
                center = random.choice(units)
            else:
                center = comcenter[0]
            enemy_units = self.get_enemy_units(obs)
            distances = self.get_distances(obs, enemy_units, (center.x, center.y))
            enemy_target = enemy_units[np.argmin(distances)]
            
            # enemy_location = self.get_enemy_units(obs)

            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            # Only get 10 marines, randomly selected
            marauders = [marauder.tag for marauder in marauders]
            random.shuffle(marauders)
            marauders = marauders[:11]
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marauders, (enemy_target.x + x_offset, enemy_target.y + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

#===============================================================================
#     def attack_marauder_all_expansion1(self, obs):
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marauders) > 0:
#             attack_xy = (18, 44) if self.base_top_left else (40, 23)
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             # Only get 10 marines, randomly selected
#             marauders = [marauder.tag for marauder in marauders]
#             random.shuffle(marauders)
#             marauders =marauders[:11]
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marauders, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
# 
#     def attack_marauder_all_expansion2(self, obs):
#         marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
#         if len(marauders) > 0:
#             attack_xy = (40, 23) if self.base_top_left else (18, 44)
#             # enemy_location = self.get_enemy_units(obs)
# 
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             # Only get 10 marines, randomly selected
#             marauders = [marauder.tag for marauder in marauders]
#             random.shuffle(marauders)
#             marauders = marauders[:11]
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marauders, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
#===============================================================================


class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        action = random.choice(self.my_actions)
        return getattr(self, action)(obs)


class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.my_actions)
        self.new_game()

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
        #if self.qtable.e_greedy < .90 and self.episodeCount%5 == 0:
            #self.qtable.increment_greedy()

        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if obs.last():
            command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
            supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
            refineries = self.get_my_units_by_type(obs, units.Terran.Refinery)
            tech_labs = self.get_my_units_by_type(obs, units.Terran.BarracksTechLab)

            minerals = obs.observation.player.minerals
            vespene = obs.observation.player.vespene

            print("LAST OBSERVATION")
            print("Writing QTable to file episode_"+str(self.episodeCount))

            # CHANGE DIRECTORY NAME
            self.qtable.q_table.to_csv(r"C:\Users\arkse\Desktop\cs175_episodes\episode_"+str(self.episodeCount)+".csv", encoding='utf-8', index=False)

            print("Writing state to file episode_"+str(self.episodeCount)+"state.txt")
            file = open(r"C:\Users\arkse\Desktop\cs175_episodes\episode_"+str(self.episodeCount)+"state.txt", "w")
            file.write("#CC #SCV #IdleSCV #SupplyDepots #Refineries #CompletedRefineries #CompletedSupplyDepots #Barrackses #CompletedBarrackses #Marines #Marauders QueuedMarines QueuedMarauders  FreeSupply CanAffordSuppyDepot CanAffordBarracks CanAffordMarine CanAffordMarauder CanAffordRefinery #Hatcheries #Drones #IdleDrones #Overlords #SpawningPools #HyrdaDen #RoachWarren #BanelingNest #Zergling #Banelings #Roaches #Hydralisk #Queens #Air\n")
            file.close()
            file = open(r"C:\Users\arkse\Desktop\cs175_episodes\episode_" + str(self.episodeCount) + "state.txt", "a")
            file.write(str(self.get_state(obs)))
            file.close()

            self.episodeCount += 1
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


        if self.previous_action is not None:
            lenarmy = len(marines) + len(marauders)
            lenwork = len(scvs)
            
            if lenwork <= 0:        # To avoid division by zero
                self.qtable.learn(self.previous_state,
                                  self.previous_action,
                                  obs.reward,
                                  'terminal' if obs.last() else state)
            else:
                self.qtable.learn(self.previous_state,
                                  self.previous_action,
                                  obs.reward + (lenarmy / lenwork),
                                  'terminal' if obs.last() else state)
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)


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
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    app.run(main)
