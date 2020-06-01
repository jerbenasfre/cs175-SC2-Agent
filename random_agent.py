import random

from agent import Agent

class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        action = random.choice(self.my_actions)
        return getattr(self, action)(obs)
