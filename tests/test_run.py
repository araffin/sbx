from sbx import TQC


def test_tqc(tmp_path):
    model = TQC(
        "MlpPolicy",
        "Pendulum-v1",
        learning_starts=100,
        verbose=1,
        buffer_size=250,
        # action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
    )
    model.learn(total_timesteps=300)
    model.save(tmp_path / "test_save.zip")
    env = model.get_env()
    model = TQC.load(tmp_path / "test_save.zip", env=env)
