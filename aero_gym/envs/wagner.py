import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
from scipy import signal
from io import StringIO
from contextlib import closing
import sys

#TODO: truncate action if it would bring the state outside the observation space
#TODO: check_env -> reset() check initialize state[1]

def compute_new_wake_element(h_dot, theta, theta_dot, wake, delta_t, U=1, c=1, a=0):
    Gamma_b = compute_Gamma_b(h_dot, theta, theta_dot, U=U, c=c, a=a)
    old_wake_influence = 0.0
    for i in range(1, len(wake)):
        x = (i + 0.5) * U * delta_t + 0.5 * c
        old_wake_influence += U * delta_t * wake[i] * math.sqrt(x + 0.5 * c) / math.sqrt(x - 0.5 * c)
    new_wake_element = math.sqrt(0.5 * U * delta_t) / math.sqrt(0.5 * U * delta_t + c) * (-Gamma_b - old_wake_influence) / (U * delta_t)
    return new_wake_element

def compute_lift(h_ddot, theta_dot, theta_ddot, wake, delta_t, rho=1, U=1, c=1, a=0):
    fy = rho * c ** 2 * math.pi / 4 * (-h_ddot + a * theta_ddot - U * theta_dot)
    for i in range(0, len(wake)):
        x = (i + 0.5) * U * delta_t + 0.5 * c
        fy += rho * U * U * delta_t * wake[i] * x / math.sqrt(x ** 2 - 0.25 * c ** 2)
    return fy

def compute_Gamma_b(h_dot, theta, theta_dot, U=1, c=1, a=0):
    return math.pi * c * (h_dot + U * theta + (0.25 * c - a) * theta_dot) 

def random_fourier_series(t, T, N):
    A = np.random.normal(0, 1, N)
    #s = A[0]/2*np.ones(len(t))
    #A = np.random.randint(-1, high=2, size=N)
    #s = 0.1*A[0]*np.ones(len(t))
    s = 0.0
    s += 0.01*sum(np.sin(2*math.pi/T*n*t)*A[n]/n for n in range(1, N))
    return s

class WagnerEnv(gym.Env):
    """
    ### Observation Space
    The observation is an `ndarray` with shape `(5 + t_wake_max / delta_t,)` where the elements correspond to the following:
    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical velocity of the wing                                               | -0.1*U | 0.1*U  | m/s     |
    |   1   | angle of the wing (pos counterclockwise)                                    | -pi/36 | pi/36  | rad     |
    |   2   | angular velocity of the wing                                                | -Inf   | Inf    | rad/s   |
    |   3   | local circulation of the wake                                               | -Inf   | Inf    | m/s     |
    |   4   | local circulation of the wake minus the part released at t                  | -Inf   | Inf    | m/s     |
    |   5   | local circulation of the wake minus the parts released at t and t - delta_t | -Inf   | Inf    | m/s     |
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |  -1   | local circulation of the wake part released at t - t_wake_max               | -Inf   | Inf    | m/s     |

    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 continuous_actions=False,
                 num_discrete_actions=3,
                 delta_t=0.1,
                 t_max=100.0,
                 h_ddot_mean=0.0,
                 h_ddot_std=1.0,
                 h_ddot_N=100,
                 h_ddot_T=100,
                 rho=1.0,
                 U=1.0,
                 c=1,
                 a=0,
                 h_ddot_prescribed=None,
                 random_ics=False,
                 random_fourier_h_ddot=False,
                 reward_type=1,
                 observe_wake=True,
                 observe_h_ddot=False,
                 lift_threshold=0.01):
        self.continuous_actions = continuous_actions
        self.num_discrete_actions = num_discrete_actions
        self.delta_t = delta_t  # The time step size of the simulation
        self.t_max = t_max
        self.t_wake_max = t_max # The maximum amount of time that a wake element is saved in the state vector since its release
        self.N_wake = int(self.t_wake_max / self.delta_t) # The number of wake elements that are kept in the state vector
        if h_ddot_prescribed is not None:
            assert len(h_ddot_prescribed) >= int(t_max / delta_t) + 1, "The prescribed vertical acceleration has not enough entries for the whole simulation (including t=0)"
        self.h_ddot_prescribed = h_ddot_prescribed
        self.h_ddot_mean = h_ddot_mean
        self.h_ddot_std = h_ddot_std
        self.h_ddot_N = h_ddot_N
        self.h_ddot_T = h_ddot_T
        self.rho = rho
        self.U = U
        self.c = c
        self.a = a
        self.random_ics = random_ics
        self.random_fourier_h_ddot = random_fourier_h_ddot
        self.reward_type = reward_type

        # Create discrete system to advance non-wake states
        A = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])
        B = np.array([
            [1, 0],
            [0, 0],
            [0, 1],
        ])
        C = np.array([
            [0, 0, 0],
        ])
        D = np.array([
            [0, 0],
        ])
        sys = signal.StateSpace(A, B, C, D)
        sys = sys.to_discrete(delta_t)
        self.A = sys.A.astype(np.float32)
        self.B = sys.B.astype(np.float32)
        
        # Set limits for state, observation, and action spaces
        self.hdot_threshold = np.inf
        self.hddot_threshold = np.inf
        #self.theta_threshold = 20 * math.pi / 180
        #self.theta_dot_threshold = 10 * math.pi / 180
        self.theta_ddot_threshold = 0.1 #10 * math.pi / 180
        #self.lift_threshold = lift_threshold
        self.theta_threshold = np.inf
        self.theta_dot_threshold = np.inf
        self.lift_threshold = np.inf

        self.observe_wake = observe_wake
        self.observe_h_ddot = observe_h_ddot

        self.state_low = np.concatenate(
            (
                np.array(
                    [
                        -0.1 * U, # vertical velocity
                        -self.theta_threshold, # angle of the wing
                        -self.theta_dot_threshold, # angular velocity of the wing
                    ]
                ),
                np.full((self.N_wake,), -np.inf)
            )
        ).astype(np.float32)

        self.state_high = np.concatenate(
            (
                np.array([self.lift_threshold]),
                np.array(
                    [
                        0.1 * U, # vertical velocity
                        self.theta_threshold, # angle of the wing
                        self.theta_dot_threshold, # angular velocity of the wing
                    ]
                ),
                np.full((self.N_wake,), np.inf)
            )
        ).astype(np.float32)

        obs_low = self.state_low[0:3]
        obs_high = self.state_high[0:3]

        if self.observe_wake:
            np.append(obs_low, self.state_low[3:])
            np.append(obs_high, self.state_high[3:])
        if self.observe_h_ddot:
            np.append(obs_low, np.float32(self.h_ddot_threshold))
            np.append(obs_high, -np.float32(self.h_ddot_threshold))

        self.observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        # We have 1 action: the angular acceleration
        if self.continuous_actions:
            self.action_space = spaces.Box(-self.theta_ddot_threshold, self.theta_ddot_threshold, (1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.discrete_action_values = self.theta_ddot_threshold * np.linspace(-1, 1, num=self.num_discrete_actions) ** 3


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
        obs = self.state[0:3]
        if self.observe_wake:
            np.append(obs, self.state[3:])
        if self.observe_h_ddot:
            np.append(obs, self.h_ddot)
        return obs

    def _get_info(self):
        return {"previous lift": self.fy, "time": self.t, "time step": self.time_step, "TimeLimit.truncated"=self.truncated}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.time_step = 0
        self.truncated = False
        self.terminated = False

        # Recreate a prescribed vertical acceleration if a random fourier signal is required
        if self.random_fourier_h_ddot:
            self.h_ddot_prescribed = self.U * random_fourier_series(np.linspace(0, self.t_max, int(self.t_max / self.delta_t) + 1), self.h_ddot_T, self.h_ddot_N) 

        # If there is no prescribed vertical acceleration, create a random vertical acceleration
        if self.h_ddot_prescribed is None:
            self.h_ddot = self.np_random.normal(self.h_ddot_mean, self.h_ddot_std)
        else:
            self.h_ddot = self.h_ddot_prescribed[self.time_step]

        self.state = np.zeros(len(self.A) + self.N_wake, dtype=np.float32)
        self.fy = np.float32(0.0)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render() 

        return observation, info

    def step(self, action):
        assert self.h_ddot is not None, "Call reset before using step method."
        assert self.state is not None, "Call reset before using step method."

        # Assign action to theta_ddot
        if self.continuous_actions:
            theta_ddot = action[0]
        else:
            theta_ddot = self.discrete_action_values[action] 

        # Compute the lift and reward
        self.fy = compute_lift(
                self.h_ddot,
                self.state[2],
                theta_ddot,
                self.state[3:],
                self.delta_t,
                rho=self.rho,
                U=self.U,
                c=self.c,
                a=self.a)

        if self.reward_type == 1:
            reward = -(self.fy ** 2) # v1
        elif self.reward_type == 2:
            reward = -(self.fy ** 2 + 0.1 * self.state[4] ** 2) # v2
        elif self.reward_type == 3:
            reward = -(self.fy ** 2 + 0.1 * self.state[4] ** 2 + 0.05 * jerk ** 2) # v3
        elif self.reward_type == 4:
            reward = -(self.fy ** 2 + 0.02 * jerk ** 2) # v4
        elif self.reward_type == 5:
            reward = -1 * (math.exp((self.fy / self.lift_threshold) ** 2) - 1) + 1 # v5
        elif self.reward_type == 6:
            reward = -10 * (self.fy ** 2) + 1 / (self.t_max / self.delta_t) # v6

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # Update kinematic states
        u = np.array([self.h_ddot, theta_ddot], dtype=np.float32)
        self.state[:3] = np.matmul(self.A, self.state[:3]) + np.dot(self.B, u)

        # Update wake states
        self.state[4:] = self.state[3:-1]
        self.state[3] = compute_new_wake_element(
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3:],
                self.delta_t,
                U=self.U,
                c=self.c,
                a=self.a)

        # If there is no prescribed vertical acceleration, create a random vertical acceleration
        if self.h_ddot_prescribed is None:
            self.h_ddot = self.np_random.normal(self.h_ddot_mean, self.h_ddot_std)
        else:
            self.h_ddot = self.h_ddot_prescribed[self.time_step]

        # Check if timelimit is reached or state went out of bounds
        if self.t > self.t_max or math.isclose(self.t, self.t_max, rel_tol=1e-9):
            self.truncated = True
        if self.state[1] < -self.theta_threshold or self.state[1] > self.theta_threshold:
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
        else:
            return self._render_frame()

    def _render_text(self):
        outfile = StringIO()
        outfile.write("{:5d}{:10.5f} ".format(self.time_step, self.t))
        outfile.write(("{:10.3e} "*4).format(
            self.state[0],
            self.h_ddot,
            self.state[1],
            self.state[2],
        ))
        with closing(outfile):
            return outfile.getvalue()
