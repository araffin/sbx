import os

from sbx.crossq import CrossQ
from sbx.ddpg import DDPG, GPUDDPG
from sbx.dqn import DQN, GPUDQN
from sbx.ppo import PPO
from sbx.sac import SAC, GPUSAC
from sbx.td3 import TD3, GPUTD3
from sbx.tqc import TQC, GPUTQC

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
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "TQC",
    "CrossQ",
    "GPUDDPG",
    "GPUDQN",
    "GPUSAC",
    "GPUTD3",
    "GPUTQC",
]
