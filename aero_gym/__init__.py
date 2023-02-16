from gymnasium.envs.registration import register

register(
    id="aero_gym/wagner-v0",
    entry_point="aero_gym.envs:WagnerEnv",
)
