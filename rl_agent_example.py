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

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

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
               #"build_engineering_bay",
               "train_marine",
               "train_scv",
               #"train_marauder",
               "attack",
               "attack_all")

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
            #distances = self.get_distances(obs, refinery, (scv.x, scv.y))
            #mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, refinery.tag)
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
            print("LOCATION CHOSEN:", location)
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

            print("BARRACK LOCATION CHOSEN :", barracks_xy)

            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self, obs):
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        command_centers.extend(self.get_my_units_by_type(obs, units.Terran.OrbitalCommand))
        scv_count = len(self.get_my_units_by_type(obs, units.Terran.SCV))
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(command_centers) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0 and scv_count < 19) :
            command_center = command_centers[0]
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
                if barrackses[0].order_length < 5 or barrackses[1] < 5:
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
                and free_supply > 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:

            # Maybe I should split attack locations and let the bot figure out where it is best to attack.
            # That is, if I don't implement the closest enemy unit attack method
            if self.base_top_left:
                locations = [(40,44), (18,44), (40, 23)] # bottom right, bottom left, top right
            else:
                locations = [(18, 23), (40, 23), (18,44)] # top left, top right, bottom left
            attack_xy = random.choice(locations)
            distances = self.get_distances(obs, marines, attack_xy)
            marine = marines[np.argmax(distances)]

            #enemy_location = self.get_enemy_units(obs)


            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack_all(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 10:
            if self.base_top_left:
                locations = [(40, 44), (18, 44), (40, 23)]  # bottom right, bottom left, top right
            else:
                locations = [(18, 23), (40, 23), (18, 44)]  # top left, top right, bottom left
            attack_xy = random.choice(locations)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            marines = [marine.tag for marine in marines]
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marines, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()



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
        super(SmartAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward,
                              'terminal' if obs.last() else state)
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)


def main(unused_argv):
    agent1 = SmartAgent()
    agent2 = sc2_env.Bot(sc2_env.Race.zerg,
                               sc2_env.Difficulty.very_easy)#RandomAgent()
    gameCount = 0
    start_time = time.time()
    try:
        while gameCount < 500:
            gameCount += 1
            print("=========================GAME : " + str(gameCount) + "=========================")
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.terran),
                            agent2],

                    agent_interface_format=features.AgentInterfaceFormat(
                        action_space=actions.ActionSpace.RAW,
                        use_raw_units=True,
                        raw_resolution=64,
                        feature_dimensions=features.Dimensions(screen=84, minimap=64)),
                    step_mul=48,
                    #game_steps_per_episode=0,
                    disable_fog=True,
                    visualize=True) as env:

                    agent1.setup(env.observation_spec(), env.action_spec())

                    timesteps = env.reset()
                    agent1.reset()

                    while True:
                        step_actions = [agent1.step(timesteps[0])]
                        if timesteps[0].last():
                            break
                        timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % (
            elapsed_time))

if __name__ == "__main__":
    app.run(main)

'''
def run_loop(agents, env, max_frames=0, max_episodes=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  total_episodes = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    agent.setup(obs_spec, act_spec)

  try:
    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1
        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, timesteps)]
        if max_frames and total_frames >= max_frames:
          return
        if timesteps[0].last():
          break
        timesteps = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
'''

'''
class BaseAgent(object):
  """A base agent to write custom scripted agents.
  It can also act as a passive agent that does nothing but no-ops.
  """

  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.obs_spec = None
    self.action_spec = None

  def setup(self, obs_spec, action_spec):
    self.obs_spec = obs_spec
    self.action_spec = action_spec

  def reset(self):
    self.episodes += 1

  def step(self, obs):
    self.steps += 1
    self.reward += obs.reward
    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
'''