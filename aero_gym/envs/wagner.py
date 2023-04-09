import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy import signal
from io import StringIO
from contextlib import closing

#TODO: use observation wrappers (with dicts) instead of observe_wake and observe_h_ddot
#TODO: check pressure equation
#TODO: assert xp's are between -c/2 and c/2

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

def compute_lift(h_ddot, alpha_dot, alpha_ddot, gamma_wake, x_wake, delta_x_wake, rho=1.0, U=1.0, c=1.0, a=0.0):
    """
    Compute the total lift (added mass + circulatory) (Eldredge2019 equation 8.159).
    """
    # Added mass force
    fy = 0.25 * rho * c ** 2 * np.pi * (-h_ddot - a * alpha_ddot + U * alpha_dot)
    # Circulatory force
    fy += rho * U * delta_x_wake * np.sum(gamma_wake * x_wake / np.sqrt(x_wake ** 2 - 0.25 * c ** 2))
    return fy

def compute_wake_pressure_diff(xp, gamma_wake, x_wake, delta_x_wake, rho=1.0, U=1.0, c=1.0):
    """
    Compute the pressure difference between the upper and lower surface (pu-pl) at x due to the wake.
    """
    p = -rho * U * delta_x_wake / np.pi * np.sum(gamma_wake * (x_wake + xp) / (np.sqrt(x_wake ** 2 - 0.25 * c ** 2) * np.sqrt(0.25 * c ** 2 - xp ** 2)))
    return p

class WagnerEnv(gym.Env):
    """
    ### State Space
    The state is an `ndarray` with shape `(2 + t_wake_max / delta_t,)` where the elements correspond to the following:
    | Index | State                                                                       | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | effective angle of attack                                                   | -pi/18 | pi/18  | rad     |
    |   1   | angular velocity of the wing                                                | -pi/18 | pi/18  | rad/s   |
    |   2   | angular velocity of the wing                                                | -Inf   | Inf    | rad/s   |
    |   3   | vortex sheet strength of the wake element released at the current time t    | -Inf   | Inf    | m/s     |
    |   4   | vortex sheet strength of the wake element released at t - delta_t           | -Inf   | Inf    | m/s     |
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |  -1   | vortex sheet strength of the wake element released at t - t_wake_max        | -Inf   | Inf    | m/s     |

    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 continuous_actions=False,
                 num_discrete_actions=3,
                 delta_t=0.1,
                 t_max=100.0,
                 h_ddot_prescribed=None,
                 h_ddot_generator=None,
                 rho=1.0,
                 U=1.0,
                 c=1,
                 a=0,
                 reward_type=1,
                 observe_wake=True,
                 observe_h_ddot=False,
                 observe_previous_lift=False,
                 observe_body_circulation=False,
                 observe_pressure=False,
                 pressure_sensor_positions=[],
                 lift_threshold=0.01,
                 alpha_ddot_threshold=0.1):
        self.continuous_actions = continuous_actions
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
        self.reward_type = reward_type

        self.alpha_eff_threshold = 10 * np.pi / 180
        self.h_ddot_threshold = np.inf
        self.alpha_dot_threshold = 10 * np.pi / 180
        self.alpha_ddot_threshold = alpha_ddot_threshold
        self.lift_threshold = lift_threshold

        self.observe_wake = observe_wake
        self.observe_h_ddot = observe_h_ddot
        self.observe_previous_lift = observe_previous_lift
        self.observe_body_circulation = observe_body_circulation
        self.observe_pressure = observe_pressure
        self.pressure_sensor_positions = np.array(pressure_sensor_positions)

        self.t_wake_max = t_max # The maximum amount of time that a wake element is saved in the state vector since its release
        self.N_wake = int(self.t_wake_max / self.delta_t) # The number of wake elements that are kept in the state vector
        self.x_wake = np.array([(i + 0.5) * U * delta_t + 0.5 * c for i in range(self.N_wake)])
        self.delta_x_wake = U * delta_t

        # States are the wing's effective AOA, the angular velocity, and the states to describe the lift.
        self.state_low = np.concatenate(
            (
                np.array(
                    [
                        -self.alpha_eff_threshold, # effective AOA
                        -self.alpha_dot_threshold, # angular velocity of the wing
                    ]
                ),
                np.full((self.N_wake,), -np.inf)
            )
        )

        self.state_high = np.concatenate(
            (
                np.array(
                    [
                        self.alpha_eff_threshold, # effective AOA
                        self.alpha_dot_threshold, # angular velocity of the wing
                    ]
                ),
                np.full((self.N_wake,), np.inf)
            )
        )

        obs_low = self.state_low[0:2]
        obs_high = self.state_high[0:2]

        if self.observe_wake:
            obs_low = np.append(obs_low, self.state_low[2:])
            obs_high = np.append(obs_high, self.state_high[2:])
        if self.observe_h_ddot:
            obs_low = np.append(obs_low, -self.h_ddot_threshold)
            obs_high = np.append(obs_high, self.h_ddot_threshold)
        if self.observe_previous_lift:
            obs_low = np.append(obs_low, -self.lift_threshold)
            obs_high = np.append(obs_high, self.lift_threshold)
        if self.observe_body_circulation:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np. append(obs_high, np.inf)
        if self.observe_pressure:
            obs_low = np.append(obs_low, np.full_like(self.pressure_sensor_positions, -np.inf))
            obs_high = np.append(obs_high, np.full_like(self.pressure_sensor_positions, np.inf))
 
        self.observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        # We have 1 action: the angular acceleration
        if self.continuous_actions:
            # Will be rescaled by the threshold
            self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.discrete_action_values = self.alpha_ddot_threshold * np.linspace(-1, 1, num=self.num_discrete_actions) ** 3

        # Create the discrete system to advance non-wake states
        A = np.array([
            [0, 1],
            [0, 0],
        ])
        B = np.array([
            [-1/U, (c/4-a)/U],
            [0, 1],
        ])
        C = np.array([
            [0, 0],
        ])
        D = np.array([
            [0, 0],
        ])
        sys = signal.StateSpace(A, B, C, D)
        sys = sys.to_discrete(delta_t)
        self.A = sys.A
        self.B = sys.B

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

    def _get_obs(self):
        obs = self.state[0:2]
        if self.observe_wake:
            obs = np.append(obs, self.state[2:])
        if self.observe_h_ddot:
            obs = np.append(obs, self.h_ddot)
        if self.observe_previous_lift:
            obs = np.append(obs, self.fy)
        if self.observe_body_circulation:
            body_circulation = -np.sum(self.state[2:]) * self.delta_x_wake
            obs = np.append(obs, body_circulation)
        if self.observe_pressure:
            pressure_measurements = [compute_wake_pressure_diff(xp, self.state[2:], self.x_wake, self.delta_x_wake, rho=self.rho, U=self.U, c=self.c) for xp in self.pressure_sensor_positions]
            obs = np.append(obs, pressure_measurements)
        return obs.astype(np.float32)

    def _get_info(self):
        return {"previous fy": self.fy, "previous alpha_ddot": self.alpha_ddot, "current alpha_dot": self.state[1], "current alpha_eff": self.state[0], "current h_ddot": self.h_ddot, "current t": self.t, "current time_step": self.time_step}

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
        if self.h_ddot_prescribed is not None:
            self.h_ddot_list = np.array(self.h_ddot_prescribed)
        elif self.h_ddot_generator is not None:
            self.h_ddot_list = np.array(self.h_ddot_generator(self))
        else:
            self.h_ddot_list = np.zeros(int(self.t_max / self.delta_t))
            print("No h_ddot provided, using zeros instead")

        self.h_ddot = self.h_ddot_list[self.time_step]
        self.state = np.zeros(len(self.A) + self.N_wake)
        self.fy = 0.0
        self.alpha_ddot = 0.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render() 

        return observation, info

    def step(self, action):
        assert self.h_ddot is not None, "Call reset before using step method."
        assert self.state is not None, "Call reset before using step method."

        # Assign action to alpha_ddot
        if self.continuous_actions:
            self.alpha_ddot = action[0] * self.alpha_ddot_threshold
        else:
            self.alpha_ddot = self.discrete_action_values[action] 

        # Compute the lift and reward
        self.fy = compute_lift(
                self.h_ddot,
                self.state[1],
                self.alpha_ddot,
                self.state[2:],
                self.x_wake,
                self.delta_x_wake,
                rho=self.rho,
                U=self.U,
                c=self.c,
                a=self.a)

        # Compute the reward
        #reward = 1 - (self.fy / self.lift_threshold) ** 2
        if self.reward_type == 1:
            reward = -(self.fy ** 2) # v1
        elif self.reward_type == 2:
            reward = -(self.fy ** 2 + 0.1 * self.alpha_ddot ** 2) # v2
        elif self.reward_type == 3:
            reward = -abs(self.fy / self.lift_threshold) + 1
        elif self.reward_type == 5:
            reward = -1 * (np.exp((self.fy / self.lift_threshold) ** 2) - 1) + 1 # v5

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # Update state
        u = np.array([self.h_ddot, self.alpha_ddot])
        self.state[:2] = np.matmul(self.A, self.state[:2]) + np.dot(self.B, u)

        # Update wake states
        self.state[3:] = self.state[2:-1]
        self.state[2] = compute_new_wake_element(
                self.state[0],
                self.state[2:],
                self.x_wake,
                self.delta_x_wake,
                U=self.U,
                c=self.c)

        # Check if timelimit is reached
        if self.t > self.t_max or np.isclose(self.t, self.t_max, rtol=1e-9):
            self.truncated = True
        else:
            self.h_ddot = self.h_ddot_list[self.time_step]

        # Check if state or lift goes out of bounds
        if self.state[0] < -self.alpha_eff_threshold or self.state[0] > self.alpha_eff_threshold:
            self.terminated = True
        if self.fy < -self.lift_threshold or self.fy > self.lift_threshold:
            self.terminated = True

        # Create observation for next state
        observation = self._get_obs()
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
        outfile.write((" {:10.3e}"*3).format(
            self.state[0],
            self.state[1],
            self.h_ddot,
        ))
        with closing(outfile):
            return outfile.getvalue()
