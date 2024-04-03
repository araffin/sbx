import os

from sbx.crossq import CrossQ
from sbx.ddpg import DDPG
from sbx.dqn import DQN
from sbx.droq import DroQ
from sbx.ppo import PPO
from sbx.sac import SAC
from sbx.td3 import TD3
from sbx.tqc import TQC
from sbx.tqcrossq import TQCrossQ

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "TQCrossQ",
    "CrossQ",
    "DDPG",
    "DQN",
    "DroQ",
    "PPO",
    "SAC",
    "TD3",
    "TQC",
]
