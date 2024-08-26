import os

from sbx.crossq import CrossQ
from sbx.ddpg import DDPG
from sbx.dqn import DQN
from sbx.ppo import PPO
from sbx.r_ppo import RPPO
from sbx.sac import SAC
from sbx.td3 import TD3
from sbx.tqc import TQC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def DroQ(*args, **kwargs):
    raise ImportError(
        "Since SBX 0.16.0, `DroQ` is now a special configuration of SAC.\n "
        "Please check the documentation for more information: "
        "https://github.com/araffin/sbx?tab=readme-ov-file#note-about-droq"
    )


__all__ = [
    "CrossQ",
    "DDPG",
    "DQN",
    "PPO",
    "RPPO"
    "SAC",
    "TD3",
    "TQC",
]
