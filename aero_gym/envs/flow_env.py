import gymnasium as gym
from gymnasium import spaces
import numpy as np
from io import StringIO
from contextlib import closing
import os

class FlowEnv(gym.Env):
    """
    ## Description

    The ViscousFlow environment is the two-dimensional, viscous, aerodynamic model for an airfoil undergoing arbitrary motions. The airfoil undergoes prescribed or random vertical accelerations and the goal is to minimize the lift variations by controlling the AOA through the angular acceleration of the airfoil.

    TODO

    ## Action space

    The action is a `numpy.ndarray(dtype=numpy.float32)` with shape `(1,)` which can take values in the range `[-1,1]` and represents the scaled angular acceleration (alpha_ddot) applied at a distance `a` from the midchord position.
    The actual angular acceleration depends on the value of the argument `alpha_ddot_scale`: `alpha_ddot = action * alpha_ddot_scale`.

    Alternatively, if `use_discrete_actions` is `True`, the action is a `numpy.int64` which can take discrete values `{0, 1, ..., N-1}`, with `N = `num_discrete_actions`, in which case the action `i` gets mapped to the angular acceleration `(2 * i  / ( N - 1 ) - 1 ) * alpha_ddot_scale`

    ## Observation Space

    TODO: explain scaling

    The default observation space is an `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | angle of attack at the current timestep                                     | -Inf   |  Inf   | rad     |
    |   1   | angular velocity of the wing at the current timestep                        | -Inf   |  Inf   | rad/T   |

    Setting the following keyword arguments to `True` will append the observation space with the following arrays (in the order that is given here):

    `observe_vorticity_field`
    `observe_vorticity_field`

    `observe_h_ddot`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical acceleration of the wing at the current timestep                   | -Inf   |  Inf   | L/T^2   |

    `observe_h_dot`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical velocity of the wing at the current timestep                       | -Inf   |  Inf   | L/T     |
    
    `observe_previous_lift`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | lift at the previous timestep (per unit depth)                              | -Inf   |  Inf   | M/T^2   |

    `observe_previous_pressure` (N = length of `pressure_sensor_positions`)
    
    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | pressure at the first pressure sensor position at the previous timestep     | -Inf   |  Inf   | M/LT^2  |
    |   .   |                                     .                                       |   .    |   .    |    .    |
    |   .   |                                     .                                       |   .    |   .    |    .    |
    |   .   |                                     .                                       |   .    |   .    |    .    |
    |  N-1  | pressure at the last pressure sensor position at the previous timestep      | -Inf   |  Inf   | M/LT^2  |

    ## Rewards

    TODO

    ## Starting State

    The episode time, kinematic states, flow states, and previous lift are all initialized to zero. The vertical acceleration is initialized to its first value.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination: Absolute value of the lift is greater than `lift_scale`
    2. Truncation: Episode time `t` is greater than or equal to `t_max`

    ## Arguments

    TODO

    """
    metadata = {"render_modes": ["ansi", "grayscale_array"], "render_fps": 4}

    def __init__(self,
                 use_discrete_actions=False,
                 num_discrete_actions=3,
                 delta_t=0.1,
                 t_max=1.0,
                 h_ddot_prescribed=None,
                 h_ddot_generator=None,
                 reference_lift_prescribed=None,
                 reference_lift_generator=None,
                 rho=1.0,
                 U=1.0,
                 c=1,
                 a=0,
                 alpha_init=0.0,
                 reward_type=3,
                 observe_h_ddot=False,
                 observe_h_dot=False,
                 observe_previous_lift=False,
                 observe_previous_lift_error=False,
                 observe_previous_pressure=False,
                 pressure_sensor_positions=[],
                 lift_termination=False,
                 lift_upper_limit=None,
                 lift_lower_limit=None,
                 h_dot_limit=1.0,
                 alpha_dot_limit=2.0,
                 alpha_upper_limit=60 * np.pi / 180,
                 alpha_lower_limit=-60 * np.pi / 180,
                 h_dot_termination=False,
                 alpha_dot_termination=False,
                 alpha_termination=False,
                 lift_scale=1.0,
                 alpha_ddot_scale=1.0,
                 alpha_dot_scale=1.0,
                 alpha_scale=1.0,
                 h_ddot_scale=1.0,
                 h_dot_scale=1.0):
        self.use_discrete_actions = use_discrete_actions
        self.num_discrete_actions = num_discrete_actions
        self.delta_t = delta_t  # The time step size of the simulation
        self.t_max = t_max
        if h_ddot_prescribed is not None:
            assert len(h_ddot_prescribed) >= int(t_max / delta_t), "The prescribed vertical acceleration has not enough entries for the whole simulation (starting at t=0)"
        self.h_ddot_prescribed = h_ddot_prescribed
        self.h_ddot_generator = h_ddot_generator
        self.reference_lift_prescribed = reference_lift_prescribed
        if reference_lift_prescribed is not None:
            assert len(reference_lift_prescribed) >= int(t_max / delta_t), "The prescribed reference lift has not enough entries for the whole simulation (starting at t=0)"
        self.reference_lift_generator = reference_lift_generator
        self.rho = rho
        self.U = U
        self.c = c
        self.a = a
        self.alpha_init = alpha_init
        self.reward_type = reward_type

        self.h_dot_scale = h_dot_scale
        self.h_ddot_scale = h_ddot_scale
        self.alpha_eff_scale = lift_scale
        self.alpha_scale = alpha_scale
        self.alpha_dot_scale = alpha_dot_scale
        self.alpha_ddot_scale = alpha_ddot_scale
        self.lift_scale = lift_scale
        self.pressure_scale = lift_scale / c
        self.lift_termination = lift_termination
        self.h_dot_termination = h_dot_termination
        self.alpha_dot_termination = alpha_dot_termination
        self.alpha_termination = alpha_termination
        self.h_dot_limit = h_dot_limit
        self.alpha_dot_limit = alpha_dot_limit
        self.alpha_upper_limit = alpha_upper_limit
        self.alpha_lower_limit = alpha_lower_limit

        self.observe_h_ddot = observe_h_ddot
        self.observe_h_dot = observe_h_dot
        self.observe_previous_lift = observe_previous_lift
        self.observe_previous_lift_error = observe_previous_lift_error
        self.observe_previous_pressure = observe_previous_pressure
        self.pressure_sensor_positions = np.array(pressure_sensor_positions)

        # Set upper and lower lift limits to lift scale if they were not provided
        if lift_upper_limit is None or lift_lower_limit is None:
            print("`lift_upper_limit` and/or `lift_lower_limit` not provided, setting limits to plus and minus `lift_scale`")
            self.lift_upper_limit = self.lift_scale
            self.lift_lower_limit = -self.lift_scale
        else:
            self.lift_upper_limit = lift_upper_limit
            self.lift_lower_limit = lift_lower_limit

        # The default observation
        obs_low = np.array(
            [
                -np.inf, # AOA
                -np.inf, # angular velocity of the wing
            ]
        )

        obs_high = np.array(
            [
                np.inf, # AOA
                np.inf, # angular velocity of the wing
            ]
        )
        
        # Extra observations specified with keyword arguments
        if self.observe_h_ddot:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift_error:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_pressure:
            obs_low = np.append(obs_low, np.full_like(self.pressure_sensor_positions, -np.inf))
            obs_high = np.append(obs_high, np.full_like(self.pressure_sensor_positions, np.inf))

        self.observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        # We have 1 action: the angular acceleration
        if self.use_discrete_actions:
            self.action_space = spaces.Discrete(self.num_discrete_actions)
        else:
            # Will be rescaled by the threshold
            self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.discrete_action_values = self.alpha_ddot_scale * np.linspace(-1, 1, num=self.num_discrete_actions) ** 3

        return self.observation_space

    def _update_kin_state_attributes(self):
        # Overload in child environment
        return

    def _get_obs(self):
        self._update_kin_state_attributes()
        scalar_obs = np.array([self.alpha / self.alpha_scale, self.alpha_dot / self.alpha_dot_scale])

        # create observation vector
        if self.observe_h_ddot:
            scalar_obs = np.append(scalar_obs, self.h_ddot / self.h_ddot_scale)
        if self.observe_h_dot:
            scalar_obs = np.append(scalar_obs, self.h_dot / self.h_dot_scale)
        if self.observe_previous_lift:
            scalar_obs = np.append(scalar_obs, self.fy / self.lift_scale)
        if self.observe_previous_lift_error:
            scalar_obs = np.append(scalar_obs, self.fy_error / self.lift_scale)
        if self.observe_previous_pressure:
            scaled_pressure = [p / self.pressure_scale for p in self.p]
            scalar_obs = np.append(scalar_obs, scaled_pressure)

        scalar_obs = scalar_obs.astype(np.float32) 

        return scalar_obs

    def _get_info(self):
        self._update_kin_state_attributes()
        return {"scaled previous fy": self.fy / self.lift_scale,
                "scaled previous fy_error": self.fy_error / self.lift_scale,
                "scaled previous alpha_ddot": self.alpha_ddot / self.alpha_ddot_scale,
                "scaled alpha_dot": self.alpha_dot / self.alpha_dot_scale,
                "scaled alpha": self.alpha / self.alpha_scale,
                "scaled h_ddot": self.h_ddot / self.h_ddot_scale,
                "scaled h_dot": self.h_dot / self.h_dot_scale,
                "scaled reference_lift": self.reference_lift / self.lift_scale,
                "unscaled previous fy": self.fy,
                "unscaled previous fy_error": self.fy_error,
                "unscaled previous alpha_ddot": self.alpha_ddot,
                "unscaled alpha_dot": self.alpha_dot,
                "unscaled alpha": self.alpha,
                "unscaled h_ddot": self.h_ddot,
                "unscaled h_dot": self.h_dot,
                "unscaled reference_lift": self.reference_lift / self.lift_scale,
                "t": self.t,
                "time_step": self.time_step}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.time_step = 0
        self.truncated = False
        self.terminated = False

        # Assign the options to the relevant fields
        if options is not None:
            if "h_ddot_prescribed" in options:
                self.h_ddot_prescribed = options["h_ddot_prescribed"]
            if "h_ddot_generator" in options:
                self.h_ddot_generator = options["h_ddot_generator"]
            if "reference_lift_prescribed" in options:
                self.reference_lift_prescribed = options["reference_lift_prescribed"]
            if "reference_lift_generator" in options:
                self.reference_lift_generator = options["reference_lift_generator"]

        # If there is no prescribed vertical acceleration use the provided function to generate one. If no function was provided, set the vertical acceleration to zero.
        if self.h_ddot_prescribed is not None:
            self.h_ddot_list = np.array(self.h_ddot_prescribed)
        elif self.h_ddot_generator is not None:
            self.h_ddot_list = np.array(self.h_ddot_generator(self))
        else:
            self.h_ddot_list = np.zeros(int(self.t_max / self.delta_t))
            print("No h_ddot provided, using zeros instead")
        assert len(self.h_ddot_list) >= int(self.t_max / self.delta_t), "The prescribed vertical acceleration has not enough entries for the whole simulation (starting at t=0)"

        # If there is no prescribed reference lift use the provided function to generate one. If no function was provided, set the reference lift to zero.
        if self.reference_lift_prescribed is not None:
            self.reference_lift_list = np.array(self.reference_lift_prescribed)
        elif self.reference_lift_generator is not None:
            self.reference_lift_list = np.array(self.reference_lift_generator(self))
        else:
            self.reference_lift_list = np.zeros(int(self.t_max / self.delta_t))
            print("No reference lift provided, using zeros instead")
        assert len(self.reference_lift_list) >= int(self.t_max / self.delta_t), "The prescribed reference lift has not enough entries for the whole simulation (starting at t=0)"

        self.h_ddot = self.h_ddot_list[self.time_step]
        self.reference_lift = self.reference_lift_list[self.time_step]
        self.alpha = self.alpha_init
        self.alpha_dot = 0.0
        self.alpha_ddot = 0.0
        self.d_alpha_ddot = 0.0
        self.h_dot = 0.0
        self.h_ddot = 0.0
        # The following should be set to their actual values in the child's reset function
        self.fy = 0.0
        self.fy_error = 0.0
        self.p = np.zeros_like(self.pressure_sensor_positions)

        return

    def _assign_action(self, action):
        # Assign action to alpha_ddot
        if self.use_discrete_actions:
            new_alpha_ddot = self.discrete_action_values[action] 
        else:
            # Clip and scale
            new_alpha_ddot = min(max(action[0], -1), 1) * self.alpha_ddot_scale

        # compute alpha_ddot differnce for possible penalization
        self.d_alpha_ddot = new_alpha_ddot - self.alpha_ddot
        self.alpha_ddot = new_alpha_ddot

    def _check_termination_truncation(self):
        # Check if timelimit is reached
        if self.t > self.t_max or np.isclose(self.t, self.t_max, rtol=1e-9):
            self.truncated = True

        # Check if lift goes out of bounds
        if self.lift_termination and (self.fy < self.lift_lower_limit or self.fy > self.lift_upper_limit):
            self.terminated = True
            # self.truncated = True

        # Check if alpha, h_dot, or alpha_dot go out of bounds
        if self.alpha_termination and (self.alpha < self.alpha_lower_limit or self.alpha > self.alpha_upper_limit):
            self.terminated = True
            # self.truncated = True
        if self.alpha_dot_termination and (self.alpha_dot < -self.alpha_dot_limit or self.alpha_dot > self.alpha_dot_limit):
            self.terminated = True
            # self.truncated = True
        if self.h_dot_termination and (self.h_dot < -self.h_dot_limit or self.h_dot > self.h_dot_limit):
            # self.terminated = True
            self.truncated = True

        return self.terminated, self.truncated

    def _compute_reward(self):
        # Compute the reward
        if self.reward_type == 1:
            reward = -((self.fy_error / self.lift_scale) ** 2) # v1
        elif self.reward_type == 2:
            reward = -((self.fy_error / self.lift_scale) ** 2 + 0.1 * (self.alpha_ddot / self.alpha_ddot_scale) ** 2) # v2
        elif self.reward_type == 3:
            reward = -abs(self.fy_error / self.lift_scale) + 1
        elif self.reward_type == 4:
            reward = -abs(self.fy_error / self.lift_scale) - 2 * abs(self.d_alpha_ddot / self.alpha_ddot_scale) + 1
            if self.terminated:
                reward -= 100
        elif self.reward_type == 5:
            reward = -abs(self.fy_error / self.lift_scale) - 2 * abs(self.d_alpha_ddot / self.alpha_ddot_scale) + 1
            if self.terminated:
                reward -= 1000
        elif self.reward_type == 6:
            reward = -abs(self.fy_error / self.lift_scale) - 2 * abs(self.d_alpha_ddot / self.alpha_ddot_scale) + 1
            if abs(self.fy_error) < 0.1 * self.lift_scale:
                reward += 10
            if self.terminated:
                reward -= 100
        else:
            raise NotImplementedError("Specified reward type is not implemented.")

        return reward

    def _update_prescribed_values(self):
        self.h_ddot = self.h_ddot_list[self.time_step]
        self.reference_lift = self.reference_lift_list[self.time_step]

    def _render_text(self):
        self._update_kin_state_attributes()
        outfile = StringIO()
        outfile.write("{:5d}{:10.5f}".format(self.time_step, self.t))
        outfile.write((" {:10.3e}" * 4).format(
            self.h_dot / self.h_dot_scale,
            self.alpha / self.alpha_scale,
            self.alpha_dot / self.alpha_dot_scale,
            self.h_ddot / self.h_ddot_scale,
        ))
        with closing(outfile):
            return outfile.getvalue()

