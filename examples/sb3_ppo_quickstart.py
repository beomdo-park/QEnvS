"""Train PPO (stable-baselines3) on qubit/StateTransfer-v0.

pip install 'qenvs[gym]' stable-baselines3
python examples/sb3_ppo_quickstart.py
"""

import numpy as np

import qenvs
from qenvs.wrappers import GymnasiumEnv


def main() -> None:
    from stable_baselines3 import PPO

    env = GymnasiumEnv(qenvs.make("qubit/StateTransfer-v0", horizon=10, dt=0.2))
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64)
    model.learn(total_timesteps=20_000)

    obs, info = env.reset(seed=0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(np.asarray(action))
    print(f"final fidelity: {float(info['fidelity']):.4f}  (pi-pulse optimum = 1.0)")


if __name__ == "__main__":
    main()
