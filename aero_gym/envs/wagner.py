import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy import signal
from io import StringIO
from contextlib import closing

#TODO: assert xp's are between -c/2 and c/2
#TODO: check scalings for alpha_dot, h_dot, alpha, and alpha_eff

def compute_new_wake_element(alpha_eff, gamma_wake, x_wake, delta_x_wake, U=1.0, c=1.0):
    """
    Compute a new wake element to satisfy the Kutta condition at the trailing edge (Eldredge2019 equation 8.156).
    """
    # Quasi-steady circulation
    qscirc = -np.pi * c * U * alpha_eff
    # Integral for the wake elements at x_wake[1:]
    old_wake_influence = np.sum((gamma_wake * np.sqrt(x_wake + 0.5 * c) / np.sqrt(x_wake - 0.5 * c) * delta_x_wake)[1:])
    # Vortex sheet strength of the element at x_wake[0]
    new_wake_element = np.sqrt(0.5 * delta_x_wake) / (np.sqrt(0.5 * delta_x_wake + c) * delta_x_wake) * (-qscirc - old_wake_influence)
    return new_wake_element

def compute_circulatory_lift(gamma_wake, x_wake, delta_x_wake, rho=1.0, U=1.0, c=1.0):
    """
    Compute the circulatory lift from discretized wake (Eldredge2019 equation 8.159).
    """
    return rho * U * delta_x_wake * np.sum(gamma_wake * x_wake / np.sqrt(x_wake ** 2 - 0.25 * c ** 2))

def compute_jones_circulatory_lift(theo_C, theo_D, alpha_eff, jones_states):
    """
    Compute the circulatory lift from the Jones approximation states and alpha_eff.
    """
    return (2 * np.pi * np.dot(theo_D, [alpha_eff]) + np.dot(theo_C, jones_states))[0]

def compute_added_mass_lift(h_ddot, alpha_dot, alpha_ddot, rho=1.0, U=1.0, c=1.0, a=0.0):
    """
    Compute the added mass lift from kinematic states (Eldredge2019 equation 8.159).
    """
    return 0.25 * rho * c ** 2 * np.pi * (-h_ddot - a * alpha_ddot + U * alpha_dot)

def compute_circulatory_pressure_diff(xp, alpha_eff, gamma_wake, x_wake, delta_x_wake, rho=1.0, U=1.0, c=1.0):
    """
    Compute the circulatory pressure difference between the upper and lower surface (pu-pl) at `xp` from the wake state.
    """
    p = np.sqrt((c/2 - xp) / (c/2 + xp)) * (-2 * rho * U ** 2 * alpha_eff + rho * U / np.pi * delta_x_wake * np.sum(gamma_wake / (np.sqrt(x_wake ** 2 - 0.25 * c ** 2))))
    return p

def compute_added_mass_pressure_diff(xp, h_ddot, alpha_dot, alpha_ddot, rho=1.0, U=1.0, c=1.0, a=0.0):
    """
    Compute the added mass pressure difference between the upper and lower surface (pu-pl) at `xp`.
    """
    return -2 * rho * (h_ddot + a * alpha_ddot - U * alpha_dot) * np.sqrt(0.25 * c ** 2 - xp ** 2)


class WagnerEnv(gym.Env):
    """
    ## Description

    The Wagner environment is the aerodynamic model for a flat plate undergoing arbitrary motions in the context of classical unsteady aerodynamics, or, the Wagner problem (Wagner, 1925). The flat plate undergoes prescribed or random vertical accelerations and the goal is to minimize the lift variations by controlling the AOA through the angular acceleration of the plate.

    This environment provides two approaches to the user:
    1. Solution through the Wagner approximation of R.T. Jones
    2. Solution through the discretization of the wake

    TODO

    ## Action space

    The action is a `numpy.ndarray(dtype=numpy.float32)` with shape `(1,)` which can take values in the range `[-1,1]` and represents the angular acceleration (alpha_ddot) applied at a distance `a` from the midchord position.
    The actual angular acceleration depends on the value of the argument `alpha_ddot_scale`: `alpha_ddot = action * alpha_ddot_scale`.

    Alternatively, if `use_discrete_actions` is `True`, the action is a `numpy.int64` which can take discrete values `{0, 1, ..., N-1}`, with `N = `num_discrete_actions`, in which case the action `i` gets mapped to the angular acceleration `alpha_ddot = (2 * i  / ( N - 1 ) - 1 ) * alpha_ddot_scale`

    The action space depends on the values of the arguments `continuous_actions`, `num_discrete_actions`, and `alpha_ddot_scale`.

    ## Observation Space

    TODO: explain scaling

    The default observation space is an `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | angle of attack at the current timestep                                     | -Inf   |  Inf   | rad     |
    |   1   | angular velocity of the wing at the current timestep                        | -Inf   |  Inf   | rad/T   |

    Setting the following keyword arguments to `True` will append the observation space with the following arrays (in the order that is given here):

    `observe_alpha_eff`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | effective angle of attack at the current timestep                           | -Inf   |  Inf   | rad     |

    `observe_previous_alpha_eff`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | effective angle of attack at the previous timestep                          | -Inf   |  Inf   | rad     |

    `observe_wake` (N = `t_max` / `delta_t`) (with `use_discretized_wake = False`)

    | Index | Observation                                                                                    | Min    | Max    | Unit    |
    |-------|------------------------------------------------------------------------------------------------|--------|--------|---------|
    |   0   | value of the first state of the R.T. Jones state-space representation at the current timestep  | -Inf   |  Inf   |  -      |
    |   1   | value of the second state of the R.T. Jones state-space representation at the current timestep | -Inf   |  Inf   |  -      |

    `observe_wake` (N = `t_max` / `delta_t`) (with `use_discretized_wake = True`)

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vortex sheet strength of the wake element released at the current timestep  | -Inf   |  Inf   | L/T     |
    |   1   | vortex sheet strength of the wake element released at the previous timestep | -Inf   |  Inf   | L/T     |
    |   .   |                                     .                                       |   .    |   .    |  .      |
    |   .   |                                     .                                       |   .    |   .    |  .      |
    |   .   |                                     .                                       |   .    |   .    |  .      |
    |  N-1  | vortex sheet strength of the wake element released at t - `t_max`           | -Inf   |  Inf   | L/T     |

    `observe_previous_wake` (N = `t_max` / `delta_t`) (with `use_discretized_wake = False`)

    | Index | Observation                                                                                     | Min    | Max    | Unit    |
    |-------|-------------------------------------------------------------------------------------------------|--------|--------|---------|
    |   0   | value of the first state of the R.T. Jones state-space representation at the previous timestep  | -Inf   |  Inf   |  -      |
    |   1   | value of the second state of the R.T. Jones state-space representation at the previous timestep | -Inf   |  Inf   |  -      |

    `observe_previous_wake` (N = `t_max` / `delta_t`) (with `use_discretized_wake = True`)

    | Index | Observation                                                                   | Min    | Max    | Unit    |
    |-------|-------------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vortex sheet strength of the wake element released at the previous timestep   | -Inf   |  Inf   | L/T     |
    |   1   | vortex sheet strength of the wake element released two timesteps earlier      | -Inf   |  Inf   | L/T     |
    |   .   |                                     .                                         |   .    |   .    |  .      |
    |   .   |                                     .                                         |   .    |   .    |  .      |
    |   .   |                                     .                                         |   .    |   .    |  .      |
    |  N-1  | vortex sheet strength of the wake element released at t - `t_max` - `delta_t` | -Inf   |  Inf   | L/T     |

    `observe_h_ddot`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical acceleration of the wing at the current timestep                   |-0.1U/dt| 0.1U/dt| L/T^2   |

    `observe_previous_lift`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | lift at the previous timestep (per unit depth)                              |see args|see args| M/T^2   |

    `observe_previous_circulatory_pressure` (N = length of `pressure_sensor_positions`)
    
    | Index | Observation                                                                         | Min    | Max    | Unit    |
    |-------|-------------------------------------------------------------------------------------|--------|--------|---------|
    |   0   | circulatory pressure at the first pressure sensor position at the previous timestep | -Inf   |  Inf   | M/LT^2  |
    |   .   |                                     .                                               |   .    |   .    |    .    |
    |   .   |                                     .                                               |   .    |   .    |    .    |
    |   .   |                                     .                                               |   .    |   .    |    .    |
    |  N-1  | circulatory pressure at the last pressure sensor position at the previous timestep  | -Inf   |  Inf   | M/LT^2  |

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

    The episode time, kinematic and wake states, and previous lift are all initialized to zero. The vertical acceleration is initialized to its first value.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination: Absolute value of the lift is greater than `lift_scale`
    2. Truncation: Episode time `t` is greater than or equal to `t_max`

    ## Arguments

    TODO

    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 use_discrete_actions=False,
                 num_discrete_actions=3,
                 delta_t=0.1,
                 t_max=100.0,
                 h_ddot_prescribed=None,
                 h_ddot_generator=None,
                 rho=1.0,
                 U=1.0,
                 c=1,
                 a=0,
                 use_discretized_wake=False,
                 reward_type=1,
                 observe_alpha_eff=False,
                 observe_previous_alpha_eff=False,
                 observe_wake=False,
                 observe_previous_wake=False,
                 observe_h_ddot=False,
                 observe_previous_lift=False,
                 observe_previous_circulatory_pressure=False,
                 observe_previous_pressure=False,
                 pressure_sensor_positions=[],
                 lift_termination=False,
                 lift_scale=0.1,
                 alpha_ddot_scale=0.1,
                 h_ddot_scale=0.05):
        self.use_discrete_actions = use_discrete_actions
        self.num_discrete_actions = num_discrete_actions
        self.delta_t = delta_t  # The time step size of the simulation
        self.t_max = t_max
        if h_ddot_prescribed is not None:
            assert len(h_ddot_prescribed) >= int(t_max / delta_t), "The prescribed vertical acceleration has not enough entries for the whole simulation (starting at t=0)"
        self.h_ddot_prescribed = h_ddot_prescribed
        self.h_ddot_generator = h_ddot_generator
        self.rho = rho
        self.U = U
        self.c = c
        self.a = a
        self.use_discretized_wake = use_discretized_wake
        self.reward_type = reward_type

        self.h_dot_scale = lift_scale * U
        self.h_ddot_scale = h_ddot_scale
        self.alpha_eff_scale = lift_scale
        self.alpha_scale = lift_scale
        self.alpha_dot_scale = h_ddot_scale / U
        self.alpha_ddot_scale = alpha_ddot_scale
        self.lift_scale = lift_scale
        self.lift_termination = lift_termination
        self.pressure_scale = lift_scale / c

        self.observe_alpha_eff = observe_alpha_eff
        self.observe_previous_alpha_eff = observe_previous_alpha_eff
        self.observe_wake = observe_wake
        self.observe_previous_wake = observe_previous_wake
        self.observe_h_ddot = observe_h_ddot
        self.observe_previous_lift = observe_previous_lift
        self.observe_previous_circulatory_pressure = observe_previous_circulatory_pressure
        self.observe_previous_pressure = observe_previous_pressure
        self.pressure_sensor_positions = np.array(pressure_sensor_positions)
        
        if self.use_discretized_wake:
            self.t_wake_max = t_max # The maximum amount of time that a wake element is saved in the state vector since its release
            self.N_wake_states = int(self.t_wake_max / self.delta_t) # The number of wake elements that are kept in the state vector
            self.x_wake = np.array([(i + 0.5) * U * delta_t + 0.5 * c for i in range(self.N_wake_states)])
            self.delta_x_wake = U * delta_t
        else:
            self.N_wake_states = 2

        # The observations don't include the vertical velocity
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
        if self.observe_h_ddot:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_circulatory_pressure:
            obs_low = np.append(obs_low, np.full_like(self.pressure_sensor_positions, -np.inf))
            obs_high = np.append(obs_high, np.full_like(self.pressure_sensor_positions, np.inf))
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
            [-1/U, (c/4-a)/U],
        ])
        C = np.array([
            [0, 0, 0, 0],
        ])
        D = np.array([
            [0, 0],
        ])
        kin_sys = signal.StateSpace(A, B, C, D)
        kin_sys = kin_sys.to_discrete(delta_t)
        self.kin_A = kin_sys.A
        self.kin_B = kin_sys.B

        # Create the discrete system to advance the Jones states
        theo_A = np.array([[-0.691, -0.0546], [1, 0]])
        theo_B = np.array([[1], [0]])
        theo_C = np.array([[0.2161, 0.0273]])
        theo_D = np.array([[0.5]])
        theo_sys = signal.StateSpace(theo_A, theo_B, theo_C, theo_D)
        theo_sys = theo_sys.to_discrete(delta_t)
        self.theo_A = theo_sys.A
        self.theo_B = theo_sys.B
        self.theo_C = theo_sys.C
        self.theo_D = theo_sys.D

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self, current_kin_state, previous_kin_state, current_wake_state, previous_wake_state):
        obs = np.array([current_kin_state[1] / self.alpha_scale, current_kin_state[2] / self.alpha_dot_scale])
        if self.observe_alpha_eff:
            obs = np.append(obs, current_kin_state[3] / self.alpha_eff_scale)
        if self.observe_previous_alpha_eff:
            obs = np.append(obs, previous_kin_state[3] / self.alpha_eff_scale)
        if self.observe_wake:
            obs = np.append(obs, current_wake_state)
        if self.observe_previous_wake:
            obs = np.append(obs, previous_wake_state)
        if self.observe_h_ddot:
            obs = np.append(obs, self.h_ddot / self.h_ddot_scale)
        if self.observe_previous_lift:
            obs = np.append(obs, self.fy / self.lift_scale)
        if self.observe_previous_circulatory_pressure:
            pressure_measurements = [compute_circulatory_pressure_diff(xp, previous_kin_state[3] / self.alpha_eff_scale, previous_wake_state, self.x_wake, self.delta_x_wake, rho=self.rho, U=self.U, c=self.c) / self.pressure_scale for xp in self.pressure_sensor_positions]
            obs = np.append(obs, pressure_measurements)
        if self.observe_previous_pressure:
            scaled_pressure = [p / self.pressure_scale for p in self.p]
            obs = np.append(obs, scaled_pressure)

        return obs.astype(np.float32)

    def _get_info(self):
        return {"previous fy": self.fy / self.lift_scale,
                "previous alpha_ddot": self.alpha_ddot / self.alpha_ddot_scale,
                "current alpha_dot": self.kin_state[2] / self.alpha_dot_scale,
                "current alpha_eff": self.kin_state[3] / self.alpha_eff_scale,
                "current h_ddot": self.h_ddot / self.h_ddot_scale,
                "current t": self.t,
                "current time_step": self.time_step}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.time_step = 0
        self.truncated = False
        self.terminated = False

        # If there is no prescribed vertical acceleration use the provided function to generate one
        if options is not None:
            if "h_ddot_prescribed" in options:
                assert len(options["h_ddot_prescribed"]) >= int(self.t_max / self.delta_t), "The prescribed vertical acceleration has not enough entries for the whole simulation (starting at t=0)"
                self.h_ddot_prescribed = options["h_ddot_prescribed"]
            if "h_ddot_generator" in options:
                self.h_ddot_generator = options["h_ddot_generator"]
        if self.h_ddot_prescribed is not None:
            self.h_ddot_list = np.array(self.h_ddot_prescribed)
        elif self.h_ddot_generator is not None:
            self.h_ddot_list = np.array(self.h_ddot_generator(self))
        else:
            self.h_ddot_list = np.zeros(int(self.t_max / self.delta_t))
            print("No h_ddot provided, using zeros instead")

        self.h_ddot = self.h_ddot_list[self.time_step] * self.h_ddot_scale
        self.kin_state = np.zeros(len(self.kin_A))
        self.wake_state = np.zeros(self.N_wake_states)
        self.fy = 0.0
        self.p = np.zeros_like(self.pressure_sensor_positions)
        self.alpha_ddot = 0.0

        observation = self._get_obs(self.kin_state, self.kin_state, self.wake_state, self.wake_state)
        info = self._get_info()

        if self.render_mode == "human":
            self.render() 

        return observation, info

    def step(self, action):
        assert self.h_ddot is not None, "Call reset before using step method."

        # Assign action to alpha_ddot
        if self.use_discrete_actions:
            self.alpha_ddot = self.discrete_action_values[action] 
        else:
            self.alpha_ddot = action[0] * self.alpha_ddot_scale

        # Compute the lift and reward
        if self.use_discretized_wake:
            self.fy = compute_circulatory_lift(
                self.wake_state,
                self.x_wake,
                self.delta_x_wake,
                rho=self.rho,
                U=self.U,
                c=self.c)
        else:
            CL = compute_jones_circulatory_lift(
                self.theo_C,
                self.theo_D,
                self.kin_state[3],
                self.wake_state)
            self.fy = 0.5 * CL * (self.U ** 2) * self.c * self.rho

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
            - 2 * self.fy / (np.pi * self.c) * np.sqrt((0.5 * self.c + xp) / (0.5 * self.c - xp)) for xp in self.pressure_sensor_positions]

        self.fy += compute_added_mass_lift(
            self.h_ddot,
            self.kin_state[2],
            self.alpha_ddot,
            rho=self.rho,
            U=self.U,
            c=self.c,
            a=self.a)

        previous_kin_state = np.copy(self.kin_state)
        previous_wake_state = np.copy(self.wake_state)

        # Compute the reward
        if self.reward_type == 1:
            reward = -(self.fy ** 2) # v1
        elif self.reward_type == 2:
            reward = -(self.fy ** 2 + 0.1 * self.alpha_ddot ** 2) # v2
        elif self.reward_type == 3:
            reward = -abs(self.fy / self.lift_scale) + 1
        elif self.reward_type == 5:
            reward = -1 * (np.exp((self.fy / self.lift_scale) ** 2) - 1) + 1 # v5
        else:
            raise NotImplementedError("Specified reward type is not implemented.")

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # Update Jones wake states (before updating kinematic states)
        if not self.use_discretized_wake:
            u = np.array([2 * np.pi * self.kin_state[3]])
            self.wake_state = np.matmul(self.theo_A, self.wake_state) + np.dot(self.theo_B, u)

        # Update kinematic states
        u = np.array([self.h_ddot, self.alpha_ddot])
        self.kin_state = np.matmul(self.kin_A, self.kin_state) + np.dot(self.kin_B, u)

        # Update non-Jones wake states
        if self.use_discretized_wake:
            self.wake_state[1:] = self.wake_state[:-1]
            self.wake_state[0] = compute_new_wake_element(
                    self.kin_state[3],
                    self.wake_state,
                    self.x_wake,
                    self.delta_x_wake,
                    U=self.U,
                    c=self.c
            )

        # Check if timelimit is reached
        if self.t > self.t_max or np.isclose(self.t, self.t_max, rtol=1e-9):
            self.truncated = True
        else:
            self.h_ddot = self.h_ddot_list[self.time_step] * self.h_ddot_scale

        # Check if lift goes out of bounds
        if self.lift_termination and (self.fy < -self.lift_scale or self.fy > self.lift_scale):
            self.terminated = True

        # Create observation for next state
        observation = self._get_obs(self.kin_state, previous_kin_state, self.wake_state, previous_wake_state)
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render() 

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()

    def _render_text(self):
        outfile = StringIO()
        outfile.write("{:5d}{:10.5f}".format(self.time_step, self.t))
        outfile.write((" {:10.3e}" * 5).format(
            self.kin_state[0] / self.h_dot_scale,
            self.kin_state[1] / self.alpha_scale,
            self.kin_state[2] / self.alpha_dot_scale,
            self.kin_state[3] / self.alpha_eff_scale,
            self.h_ddot / self.h_ddot_scale,
        ))
        with closing(outfile):
            return outfile.getvalue()
