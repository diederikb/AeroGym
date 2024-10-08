import gymnasium as gym
from gymnasium import spaces
import numpy as np
from io import StringIO
from contextlib import closing
import logging

# TODO: reverse error definition

class FlowEnv(gym.Env):
    r"""
    ## Description

    The base class for different flow environments that implement the Gymnasium environment API. This class implements some of the functionality that is the same for all flow environments.

    ## Action space

    The action is a `numpy.ndarray(dtype=numpy.float32)` with shape `(1,)` which can take values in the range `[-1,1]` and represents the scaled angular acceleration (alpha_ddot) applied at a distance `a` from the midchord position.
    The actual angular acceleration depends on the value of the argument `alpha_ddot_scale`: `alpha_ddot = action * alpha_ddot_scale`.

    Alternatively, if `use_discrete_actions` is `True`, the action is a `numpy.int64` which can take discrete values `{0, 1, ..., N-1}`, with `N = `num_discrete_actions`, in which case the action `i` gets mapped to the angular acceleration `(2 * i  / ( N - 1 ) - 1 ) * alpha_ddot_scale`

    ## Observation Space

    The observation space depends on the subclass implementation. This superclass constructor and _get_obs() methods can create an observation space and vector, which the subclass can further modify. Each element of the observation vector is scaled by its corresponding scaling, which can be passed to the constructor. For example, the angular velocity `alpha_dot` is scaled by `alpha_dot_scaled`. This allows the user to normalize the observations to improve the neural network training.

    The observation space is an `ndarray`. Setting the following keyword arguments to `True` will append the observation space with the following arrays (in the order that is given here):

    `observe_alpha` (True by default)

    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | latest available angle of attack at the current timestep                    | -Inf   |  Inf   |

    `observe_alpha_dot` (True by default)

    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | latest available angular velocity of the wing at the current timestep       | -Inf   |  Inf   |

    `observe_alpha_ddot`

    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | the previous angular acceleration of the wing                               | -Inf   |  Inf   |

    `observe_h_ddot`

    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | vertical acceleration of the wing at the current timestep                   | -Inf   |  Inf   |

    `observe_h_dot`

    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | vertical velocity of the wing at the current timestep                       | -Inf   |  Inf   |
    
    `observe_previous_lift`

    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | lift at the previous timestep                                               | -Inf   |  Inf   |
    
    `observe_previous_lift_error`

    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | lift minus reference lift at the previous timestep                          | -Inf   |  Inf   |

    `observe_previous_pressure` (N = length of `pressure_sensor_positions`)
    
    | Index | Observation                                                                 | Min    | Max    |
    |-------|-----------------------------------------------------------------------------|--------|--------|
    |   0   | pressure at the first pressure sensor position at the previous timestep     | -Inf   |  Inf   |
    |   .   |                                     .                                       |   .    |   .    |
    |   .   |                                     .                                       |   .    |   .    |
    |   .   |                                     .                                       |   .    |   .    |
    |  N-1  | pressure at the last pressure sensor position at the previous timestep      | -Inf   |  Inf   |

    ## Rewards

    The reward function can be specified by setting the `reward_type` keyword to the corresponding number of the desired function. Check the implementation of `_compute_reward()` for available options.

    ## Starting State

    The vertical acceleration is initialized to its first value. The angle of attack is initialized to `alpha_init`. The other kinematic states, the episode time, flow states, and previous lift are all initialized to zero.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination:
        occurs when either of the following is true:
        - `lift_termination` is `True` and the lift is higher than `lift_upper_limit` or lower than `lift_lower_limit`.
        - `alpha_termination` is `True` and `alpha` is higher than `alpha_upper_limit` or lower than `alpha_lower_limit`.
        - `alpha_dot_termination` is `True` and `alpha_dot` is higher than `alpha_dolt_limit` or lower than `-alpha_dot_limit`.
        - `h_dot_termination` is `True` and `h_dot` is higher than `h_dot_limit` or lower than `-h_dot_limit`.
    2. Truncation: Episode time `t` is greater than or equal to `t_max`

    ## Arguments

    TODO

    """
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
                 observe_alpha=True,
                 observe_alpha_dot=True,
                 observe_alpha_ddot=False,
                 observe_h_ddot=False,
                 observe_h_dot=False,
                 observe_previous_lift=False,
                 observe_previous_lift_error=False,
                 observe_previous_lift_integrated_error=False,
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

        self.observe_alpha = observe_alpha
        self.observe_alpha_dot = observe_alpha_dot
        self.observe_alpha_ddot = observe_alpha_ddot
        self.observe_h_ddot = observe_h_ddot
        self.observe_h_dot = observe_h_dot
        self.observe_previous_lift = observe_previous_lift
        self.observe_previous_lift_error = observe_previous_lift_error
        self.observe_previous_lift_integrated_error = observe_previous_lift_integrated_error
        self.observe_previous_pressure = observe_previous_pressure
        self.pressure_sensor_positions = np.array(pressure_sensor_positions)

        # Set upper and lower lift limits to lift scale if they were not provided
        if lift_upper_limit is None or lift_lower_limit is None:
            logging.info("`lift_upper_limit` and/or `lift_lower_limit` not provided, setting limits to plus and minus `lift_scale`")
            self.lift_upper_limit = self.lift_scale
            self.lift_lower_limit = -self.lift_scale
        else:
            self.lift_upper_limit = lift_upper_limit
            self.lift_lower_limit = lift_lower_limit

        # The default observation
        obs_low = np.array(
            [
                # No default observation
            ]
        )

        obs_high = np.array(
            [
                # No default observation
            ]
        )
        
        # Observations specified with keyword arguments
        if self.observe_alpha:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_alpha_dot:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_alpha_ddot:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_h_ddot:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift_error:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift_integrated_error:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_pressure:
            obs_low = np.append(obs_low, np.full_like(self.pressure_sensor_positions, -np.inf))
            obs_high = np.append(obs_high, np.full_like(self.pressure_sensor_positions, np.inf))

        self.scalar_observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        # We have 1 action: the angular acceleration
        if self.use_discrete_actions:
            self.action_space = spaces.Discrete(self.num_discrete_actions)
        else:
            # Will be rescaled by the threshold
            self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.discrete_action_values = self.alpha_ddot_scale * np.linspace(-1, 1, num=self.num_discrete_actions) ** 3

    def _update_kin_state_attributes(self):
        # Overload in child environment
        return

    def _update_hist(self):
        self.t_hist = np.append(self.t_hist, self.t)
        self.h_dot_hist = np.append(self.h_dot_hist, self.h_dot)
        self.h_ddot_hist = np.append(self.h_ddot_hist, self.h_ddot)
        self.alpha_hist = np.append(self.alpha_hist, self.alpha)
        self.alpha_dot_hist = np.append(self.alpha_dot_hist, self.alpha_dot)
        self.alpha_ddot_hist = np.append(self.alpha_ddot_hist, self.alpha_ddot)
        self.fy_hist = np.append(self.fy_hist, self.fy)
        self.reference_lift_hist = np.append(self.reference_lift_hist, self.reference_lift)

    def _get_obs(self):
        self._update_kin_state_attributes()
        scalar_obs = np.array([])

        # create observation vector
        if self.observe_alpha:
            scalar_obs = np.append(scalar_obs, self.alpha / self.alpha_scale)
        if self.observe_alpha_dot:
            scalar_obs = np.append(scalar_obs, self.alpha_dot / self.alpha_dot_scale)
        if self.observe_alpha_ddot:
            scalar_obs = np.append(scalar_obs, self.alpha_ddot / self.alpha_ddot_scale)
        if self.observe_h_ddot:
            scalar_obs = np.append(scalar_obs, self.h_ddot / self.h_ddot_scale)
        if self.observe_h_dot:
            scalar_obs = np.append(scalar_obs, self.h_dot / self.h_dot_scale)
        if self.observe_previous_lift:
            scalar_obs = np.append(scalar_obs, self.fy / self.lift_scale)
        if self.observe_previous_lift_error:
            scalar_obs = np.append(scalar_obs, self.fy_error / self.lift_scale)
        if self.observe_previous_lift_integrated_error:
            scalar_obs = np.append(scalar_obs, self.fy_integrated_error / self.lift_scale)
        if self.observe_previous_pressure:
            scaled_pressure = [p / self.pressure_scale for p in self.p]
            scalar_obs = np.append(scalar_obs, scaled_pressure)

        scalar_obs = scalar_obs.astype(np.float32) 

        return scalar_obs

    def _get_info(self):
        self._update_kin_state_attributes()
        return {
            "t_hist": self.t_hist,
            "h_dot_hist": self.h_dot_hist,
            "h_ddot_hist": self.h_ddot_hist,
            "alpha_hist": self.alpha_hist,
            "alpha_dot_hist": self.alpha_dot_hist,
            "alpha_ddot_hist": self.alpha_ddot_hist,
            "fy_hist": self.fy_hist,
            "reference_lift_hist": self.reference_lift_hist,
            "t": self.t,
            "time_step": self.time_step
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.time_step = 0
        self.truncated = False
        self.terminated = False

        # Histories to be returned in info dict
        self.t_hist = np.array([])
        self.h_dot_hist = np.array([])
        self.h_ddot_hist = np.array([])
        self.alpha_hist = np.array([])
        self.alpha_dot_hist = np.array([])
        self.alpha_ddot_hist = np.array([])
        self.fy_hist = np.array([])
        self.reference_lift_hist = np.array([])

        # Assign the options to the relevant fields
        # This can be changed to looping over the option keys and trying them as attributes of self
        if options is not None:
            if "h_ddot_prescribed" in options:
                self.h_ddot_prescribed = options["h_ddot_prescribed"]
            if "h_ddot_generator" in options:
                self.h_ddot_generator = options["h_ddot_generator"]
            if "reference_lift_prescribed" in options:
                self.reference_lift_prescribed = options["reference_lift_prescribed"]
            if "reference_lift_generator" in options:
                self.reference_lift_generator = options["reference_lift_generator"]
            if "reward_type" in options:
                self.reward_type = options["reward_type"]
            if "lift_termination" in options:
                self.lift_termination = options["lift_termination"]
            if "lift_upper_limit" in options:
                self.lift_upper_limit = options["lift_upper_limit"]
            if "lift_lower_limit" in options:
                self.lift_lower_limit = options["lift_lower_limit"]

        # If there is no prescribed vertical acceleration use the provided function to generate one. If no function was provided, set the vertical acceleration to zero.
        if self.h_ddot_prescribed is not None:
            self.h_ddot_list = np.array(self.h_ddot_prescribed)
        elif self.h_ddot_generator is not None:
            self.h_ddot_list = np.array(self.h_ddot_generator(self))
        else:
            self.h_ddot_list = np.zeros(int(self.t_max / self.delta_t) + 1)
            logging.info("No h_ddot provided, using zeros instead")
        assert len(self.h_ddot_list) >= int(self.t_max / self.delta_t) + 1, "The prescribed vertical acceleration has not enough entries for the whole simulation (starting at t=0)"

        # If there is no prescribed reference lift use the provided function to generate one. If no function was provided, set the reference lift to zero.
        if self.reference_lift_prescribed is not None:
            self.reference_lift_list = np.array(self.reference_lift_prescribed)
        elif self.reference_lift_generator is not None:
            self.reference_lift_list = np.array(self.reference_lift_generator(self))
        else:
            self.reference_lift_list = np.zeros(int(self.t_max / self.delta_t) + 1)
            logging.info("No reference lift provided, using zeros instead")
        assert len(self.reference_lift_list) >= int(self.t_max / self.delta_t) + 1, "The prescribed reference lift has not enough entries for the whole simulation (starting at t=0)"

        self.h_ddot = self.h_ddot_list[self.time_step]
        self.reference_lift = self.reference_lift_list[self.time_step]
        self.alpha = self.alpha_init
        self.alpha_dot = 0.0
        self.alpha_ddot = 0.0
        self.d_alpha_ddot = 0.0
        self.h_dot = 0.0
        # The following should be set to their actual values in the child's reset function
        self.fy = 0.0
        self.fy_error = 0.0
        self.fy_integrated_error = 0.0
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
            reward = -abs(self.fy_error / self.lift_scale)
        elif self.reward_type == 3:
            reward = -abs(self.fy_error / self.lift_scale) + 1
        elif self.reward_type == 4:
            reward = -abs(self.fy_error / self.lift_scale) - 2 * abs(self.d_alpha_ddot / self.alpha_ddot_scale) + 1
            if self.terminated:
                reward -= 100
        elif self.reward_type == 5:
            reward = -abs(self.fy_error / self.lift_scale) + 1
            if self.terminated:
                reward -= 100
        elif self.reward_type == 6:
            reward = -abs(self.fy_error / self.lift_scale) - 2 * abs(self.d_alpha_ddot / self.alpha_ddot_scale) + 1
            if abs(self.fy_error) < 0.1 * self.lift_scale:
                reward += 10
            if self.terminated:
                reward -= 100
        elif self.reward_type == 7:
            reward = -abs(self.fy_error / self.lift_scale) - abs(self.fy_integrated_error) + 1
        elif self.reward_type == 8:
            reward = -np.sqrt(abs(self.fy_error / self.lift_scale)) + 1
        elif self.reward_type == 9:
            reward = -((self.fy_error / self.lift_scale) ** 2)
            if abs(self.fy_error) < 0.01 * self.lift_scale:
                reward += 10
        elif self.reward_type == 10:
            reward = -((self.fy_error / self.lift_scale) ** 2)
            if abs(self.fy_error) < 0.001 * self.lift_scale:
                reward += 10
        elif self.reward_type == 11:
            reward = -abs(self.fy_error / self.lift_scale)
            if abs(self.fy_error) < 0.001 * self.lift_scale:
                reward += 10
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

