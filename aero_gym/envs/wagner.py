from aero_gym.envs.flow_env import FlowEnv
from gymnasium import spaces
import numpy as np
from scipy import signal

#TODO: assert xp's are between -c/2 and c/2

def compute_added_mass_pressure_diff(xp, h_ddot, alpha_dot, alpha_ddot, rho=1.0, U=1.0, c=1.0, a=0.0):
    """
    Compute the added mass pressure difference between the upper and lower surface (pu-pl) at `xp`.
    """
    return -2 * rho * (h_ddot + a * alpha_ddot - U * alpha_dot) * np.sqrt(0.25 * c ** 2 - xp ** 2)


class WagnerEnv(FlowEnv):
    """
    ## Description

    This environment is the aerodynamic model for a flat plate undergoing arbitrary motions in the context of classical unsteady aerodynamics, or, the Wagner problem (Wagner, 1925). The flat plate undergoes prescribed or random vertical accelerations and the goal is to minimize the lift variations by controlling the AOA through the angular acceleration of the plate.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 observe_alpha_eff=False,
                 observe_previous_alpha_eff=False,
                 observe_wake=False,
                 observe_previous_wake=False,
                 alpha_eff_scale=1,
		 **kwargs):
        super().__init__(**kwargs)
        self.alpha_eff_scale = alpha_eff_scale
        self.observe_alpha_eff = observe_alpha_eff
        self.observe_previous_alpha_eff = observe_previous_alpha_eff
        self.observe_wake = observe_wake
        self.observe_previous_wake = observe_previous_wake
        
        self.N_wake_states = 2

        # Append the observation space of the parent class
        obs_low = self.scalar_observation_space.low
        obs_high = self.scalar_observation_space.high

        if self.observe_alpha_eff:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_alpha_eff:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_wake:
            obs_low = np.append(obs_low, np.full(self.N_wake_states, -np.inf))
            obs_high = np.append(obs_high, np.full(self.N_wake_states, np.inf))
        if self.observe_previous_wake:
            obs_low = np.append(obs_low, np.full(self.N_wake_states, -np.inf))
            obs_high = np.append(obs_high, np.full(self.N_wake_states, np.inf))
 
        self.observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        # Create the discrete system to advance non-wake states
        A = np.array([
            [0, 0, 0, 0], # h_dot
            [0, 0, 1, 0], # alpha
            [0, 0, 0, 0], # alpha_dot
            [0, 0, 1, 0], # alpha_eff
        ])
        B = np.array([
            [1, 0],
            [0, 0],
            [0, 1],
            [-1 / self.U, (self.c / 4 - self.a) / self.U],
        ])
        C = np.array([
            [0, 0, 0, 0],
        ])
        D = np.array([
            [0, 0],
        ])
        self.kin_A, self.kin_B, _, _, _ = signal.cont2discrete((A, B, C, D), self.delta_t)

        # Create the discrete system to advance the Jones states
        theo_A = np.array([[-0.691, -0.0546], [1, 0]])
        theo_B = np.array([[1], [0]])
        theo_C = np.array([[0.2161, 0.0273]])
        theo_D = np.array([[0.5]])
        self.theo_A, self.theo_B, self.theo_C, self.theo_D, _ = signal.cont2discrete((theo_A, theo_B, theo_C, theo_D), self.delta_t)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _update_kin_state_attributes(self):
        self.h_dot = self.kin_state[0]
        self.alpha = self.kin_state[1]
        self.alpha_dot = self.kin_state[2]
        return

    def _get_obs(self, current_kin_state, previous_kin_state, current_wake_state, previous_wake_state):
        obs = super()._get_obs() 
        if self.observe_alpha_eff:
            obs = np.append(obs, current_kin_state[3] / self.alpha_eff_scale)
        if self.observe_previous_alpha_eff:
            obs = np.append(obs, previous_kin_state[3] / self.alpha_eff_scale)
        if self.observe_wake:
            obs = np.append(obs, current_wake_state)
        if self.observe_previous_wake:
            obs = np.append(obs, previous_wake_state)

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.kin_state = np.array([0.0, self.alpha_init, 0.0, self.alpha_init])
        self.wake_state = np.zeros(self.N_wake_states)

        self.fy = np.pi * self.alpha_init * (self.U ** 2) * self.c * self.rho
        self.fy_error = self.reference_lift - self.fy
        self.p = [- 2 * self.fy / (np.pi * self.c) * np.sqrt((0.5 * self.c + xp) / (0.5 * self.c - xp)) for xp in self.pressure_sensor_positions]

        observation = self._get_obs(self.kin_state, self.kin_state, self.wake_state, self.wake_state)
        info = super()._get_info()

        return observation, info

    def step(self, action):
        assert self.h_ddot is not None, "Call reset before using step method."

        # Assign action to alpha_ddot and compute d_alpha_ddot
        super()._assign_action(action)

        # Compute the lift
        fy_circ = 0.5 * (self.U ** 2) * self.c * self.rho * (2 * np.pi * np.dot(self.theo_D, [self.kin_state[3]]) + np.dot(self.theo_C, self.wake_state))[0]
        fy_am = 0.25 * self.rho * self.c ** 2 * np.pi * (-self.h_ddot - self.a * self.alpha_ddot + self.U * self.kin_state[2])
        self.fy = fy_circ + fy_am
        self.fy_error = self.reference_lift - self.fy
        self.fy_integrated_error = self.fy_integrated_error + self.fy_error * self.delta_t

        # Compute the pressure using the circulatory lift
        self.p = [
            compute_added_mass_pressure_diff(
                xp,
                self.h_ddot,
                self.kin_state[2],
                self.alpha_ddot,
                rho=self.rho,
                U=self.U,
                c=self.c,
                a=self.a)
            - 2 * fy_circ / (np.pi * self.c) * np.sqrt((0.5 * self.c + xp) / (0.5 * self.c - xp)) for xp in self.pressure_sensor_positions]

        # Save the state before updating the new current state
        previous_kin_state = np.copy(self.kin_state)
        previous_wake_state = np.copy(self.wake_state)

        # Update histories
        super()._update_hist()

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # Update Jones wake states (before updating kinematic states)
        u = np.array([2 * np.pi * self.kin_state[3]])
        self.wake_state = np.matmul(self.theo_A, self.wake_state) + np.dot(self.theo_B, u)

        # Update kinematic states
        u = np.array([self.h_ddot, self.alpha_ddot])
        self.kin_state = np.matmul(self.kin_A, self.kin_state) + np.dot(self.kin_B, u)

        # Check if termination or truncation condiations are met
        terminated, truncated = super()._check_termination_truncation()

        # Compute the reward
        reward = super()._compute_reward()

        # Update prescribed (or randomly-generated) values
        super()._update_prescribed_values()

        # Create observation for next state
        observation = self._get_obs(self.kin_state, previous_kin_state, self.wake_state, previous_wake_state)
        info = super()._get_info()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return super()._render_text()
