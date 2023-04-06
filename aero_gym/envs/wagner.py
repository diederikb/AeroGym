import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
from scipy import signal
from io import StringIO
from contextlib import closing

#TODO: use observation wrappers (with dicts) instead of observe_wake and observe_h_ddot

def compute_new_wake_element(alpha_eff, wake, delta_t, U=1, c=1):
    """
    Compute a new wake element to satisfy the Kutta condition at the trailing edge (Eldredge2019 equation 8.156).
    """
    Gamma_b = -math.pi * c * U * alpha_eff
    old_wake_influence = 0.0
    for i in range(1, len(wake)):
        x = (i + 0.5) * U * delta_t + 0.5 * c
        delta_x = U * delta_t
        old_wake_influence += U * wake[i] * math.sqrt(x + 0.5 * c) / math.sqrt(x - 0.5 * c) * delta_x
    new_wake_element = math.sqrt(0.5 * U * delta_t) / math.sqrt(0.5 * U * delta_t + c) * (-Gamma_b - old_wake_influence) / (U * delta_t)
    return new_wake_element

def compute_lift(h_ddot, alpha_dot, alpha_ddot, wake, delta_t, rho=1, U=1, c=1, a=0):
    """
    Compute the total lift (added mass + circulatory) (Eldredge2019 equation 8.159).
    """
    fy = rho * c ** 2 * math.pi / 4 * (-h_ddot - a * alpha_ddot + U * alpha_dot)
    for i in range(0, len(wake)):
        x = (i + 0.5) * U * delta_t + 0.5 * c
        delta_x = U * delta_t
        fy += rho * U * wake[i] * x / math.sqrt(x ** 2 - 0.25 * c ** 2) * delta_x
    return fy

class WagnerEnv(gym.Env):
    """
    ### Observation Space
    The observation is an `ndarray` with shape `(2 + t_wake_max / delta_t,)` where the elements correspond to the following:
    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | effective angle of attack                                                   | -pi/18 | pi/18  | rad     |
    |   1   | angular velocity of the wing                                                | -pi/18 | pi/18  | rad/s   |
    |   2   | angular velocity of the wing                                                | -Inf   | Inf    | rad/s   |
    |   3   | local circulation of the wake element released at the current time t        | -Inf   | Inf    | m/s     |
    |   4   | local circulation of the wake element released at t - delta_t               | -Inf   | Inf    | m/s     |
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |  -1   | local circulation of the wake element released at t - t_wake_max            | -Inf   | Inf    | m/s     |

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
                 lift_threshold=0.01,
                 alpha_ddot_threshold=0.1):
        self.continuous_actions = continuous_actions
        self.num_discrete_actions = num_discrete_actions
        self.delta_t = delta_t  # The time step size of the simulation
        self.t_max = t_max
        if h_ddot_prescribed is not None:
            assert len(h_ddot_prescribed) >= int(t_max / delta_t), "The prescribed vertical acceleration has not enough entries for the whole simulation (including t=0)"
        self.h_ddot_prescribed = h_ddot_prescribed
        self.h_ddot_generator = h_ddot_generator
        self.rho = rho
        self.U = U
        self.c = c
        self.a = a
        self.reward_type = reward_type

        self.alpha_eff_threshold = 10 * math.pi / 180
        self.h_ddot_threshold = np.inf
        self.alpha_dot_threshold = 10 * math.pi / 180
        self.alpha_ddot_threshold = alpha_ddot_threshold
        self.lift_threshold = lift_threshold

        self.observe_wake = observe_wake
        self.observe_h_ddot = observe_h_ddot
        self.observe_previous_lift = observe_previous_lift

        self.t_wake_max = t_max # The maximum amount of time that a wake element is saved in the state vector since its release
        self.N_wake = int(self.t_wake_max / self.delta_t) # The number of wake elements that are kept in the state vector

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
        ).astype(np.float32)

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
        ).astype(np.float32)

        obs_low = self.state_low[0:2]
        obs_high = self.state_high[0:2]

        if self.observe_wake:
            obs_low = np.append(obs_low, self.state_low[2:])
            obs_high = np.append(obs_high, self.state_high[2:])
        if self.observe_h_ddot:
            obs_low = np.append(obs_low, -np.float32(self.h_ddot_threshold))
            obs_high = np.append(obs_high, np.float32(self.h_ddot_threshold))
        if self.observe_previous_lift:
            obs_low = np.append(obs_low, -np.float32(self.lift_threshold))
            obs_high = np.append(obs_high, np.float32(self.lift_threshold))

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
        self.A = sys.A.astype(np.float32)
        self.B = sys.B.astype(np.float32)

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
        return obs

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
            self.h_ddot_list = np.array(self.h_ddot_prescribed, dtype=np.float32)
        elif self.h_ddot_generator is not None:
            self.h_ddot_list = np.array(self.h_ddot_generator(self), dtype=np.float32)
        else:
            self.h_ddot_list = np.zeros(int(self.t_max / self.delta_t), dtype=np.float32)
            print("No h_ddot provided, using zeros instead")

        self.h_ddot = self.h_ddot_list[self.time_step]
        self.state = np.zeros(len(self.A) + self.N_wake, dtype=np.float32)
        self.fy = np.float32(0.0)
        self.alpha_ddot = np.float32(0.0)

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
                self.delta_t,
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
            reward = -1 * (math.exp((self.fy / self.lift_threshold) ** 2) - 1) + 1 # v5

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # Update state
        u = np.array([self.h_ddot, self.alpha_ddot], dtype=np.float32)
        self.state[:2] = np.matmul(self.A, self.state[:2]) + np.dot(self.B, u)

        # Update wake states
        self.state[3:] = self.state[2:-1]
        self.state[2] = compute_new_wake_element(
                self.state[0],
                self.state[2:],
                self.delta_t,
                U=self.U,
                c=self.c)

        # Check if timelimit is reached
        if self.t > self.t_max or math.isclose(self.t, self.t_max, rel_tol=1e-9):
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
