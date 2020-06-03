import numpy as np
import matplotlib.pyplot as plt
import random

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

import helper

import os
import pickle

class Agent(base_agent.BaseAgent):
    my_actions = ("do_nothing",
                  "harvest_minerals",
                  "harvest_vespene",
                  "build_refinery",
                  "build_supply_depot",
                  "build_barracks",
                  "build_tech_lab",
                  "train_marine",
                  "train_scv",
                  "train_marauder",
                  "attack",
                  "attack_all",
                  "attack_marine",
                  "attack_marine_all",
                  "attack_marauder",
                  "attack_marauder_all")

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
                myunits = marines + marauders
                center = random.choice(myunits)
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

    def attack_all(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        if len(marines) > 0 or len(marauders) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                myunits = marines + marauders
                center = random.choice(myunits)
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

    def attack_marine(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                myunits = marines
                center = random.choice(myunits)
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

    def attack_marine_all(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                myunits = marines
                center = random.choice(myunits)
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

    def attack_marauder(self, obs):
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        if len(marauders) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                myunits = marauders
                center = random.choice(myunits)
            else:
                center = comcenter[0]
            enemy_units = [unit for unit in obs.observation.raw_units
                     if (unit.unit_type != units.Zerg.Mutalisk
                         or unit.unit_type != units.Zerg.Corruptor
                         or unit.unit_type != units.Zerg.BroodLord
                         or unit.unit_type != units.Zerg.Overlord
                         or unit.unit_type != units.Zerg.Overseer)
                     and unit.alliance == features.PlayerRelative.ENEMY]
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

    def attack_marauder_all(self, obs):
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        if len(marauders) > 0:
            comcenter = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if len(comcenter) <= 0:
                # Get a random unit, since the command center is down
                myunits = marauders
                center = random.choice(myunits)
            else:
                center = comcenter[0]
            enemy_units = [unit for unit in obs.observation.raw_units
                           if (unit.unit_type != units.Zerg.Mutalisk
                               or unit.unit_type != units.Zerg.Corruptor
                               or unit.unit_type != units.Zerg.BroodLord
                               or unit.unit_type != units.Zerg.Overlord
                               or unit.unit_type != units.Zerg.Overseer)
                           and unit.alliance == features.PlayerRelative.ENEMY]
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
            new_addition = previous_addition + 1 / self.episodes * \
                           (current_game_result - previous_addition) 
            
            self.match_history = np.hstack((self.match_history, new_addition))
            print("Successfully updated match history.")

        except Exception as e:
            print(f"{e}\n")
            self.match_history = np.zeros((3, 1))
            self.match_history[index, 0] += 1

    def plot_match_history(self):
        x = [i for i in range(1, self.episodes + 1)]
        
        fig, ax = plt.subplots()

        ax.plot(x, self.match_history[0])
        ax.plot(x, self.match_history[1])
        ax.plot(x, self.match_history[2])

        ax.set_title("Match History")
        ax.legend(["Lose Rate", "Stalemate Rate", "Win Rate"])
        
        ax.xaxis.set_label_text("Number of Games Played")
        ax.yaxis.set_label_text("Percentage")

        plt.show()
            

    
    def save_match_history(self, folder, file_name):
        destination = helper.get_file_path(folder, file_name)
        print(f"Writing Match History to {destination}.\n")
        with open(destination, "wb") as f:
            np.save(f, self.match_history)
        
    def load_match_history(self, folder, file_name):
        try:
            target = os.path.join(folder, file_name)
            print(f"Attempting to load Match History from '{target}'.")
            with open(target, "rb") as f:
                self.match_history = np.load(f)
            print("Succeeded in loading Match History.\n")
        except Exception as e:
            print(f"{e}\n")

    def save_episode_count(self, folder, file_name):
        destination = helper.get_file_path(folder, file_name)
        print(f"Saving Episode Count to {destination}.\n")
        with open(destination, "wb") as f:
            pickle.dump(self.episodes, f)
    
    def load_episode_count(self, folder, file_name):
        try:
            target = os.path.join(folder, file_name)
            print(f"Attempting to load Episode Count from '{target}'.")
            with open(target, "rb") as f:
                self.episodes = pickle.load(f)
            print("Succeeded in loading Episode Count.\n")
        except Exception as e:
            print(f"{e}\n")