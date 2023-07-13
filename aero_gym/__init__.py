from gymnasium.envs.registration import register

register(
    id="aero_gym/wagner-v0",
    entry_point="aero_gym.envs:WagnerEnv",
)

register(
    id="aero_gym/viscous_flow-v0",
    entry_point="aero_gym.envs:ViscousFlowEnv",
)
