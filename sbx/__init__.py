import os

from sbx.dqn import DQN
from sbx.droq import DroQ
from sbx.ppo import PPO
from sbx.sac import SAC
from sbx.tqc import TQC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()
