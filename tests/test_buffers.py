import pytest
import torch as th

from sbx import SAC


def test_force_cpu_device(tmp_path):
    if not th.cuda.is_available():
        pytest.skip("No CUDA device")
    model = SAC("MlpPolicy", "Pendulum-v1", buffer_size=200)
    assert model.replay_buffer.device == th.device("cuda")
    model.save_replay_buffer(tmp_path / "replay")
    model.load_replay_buffer(tmp_path / "replay")
    assert model.replay_buffer.device == th.device("cpu")
