import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
from scipy import signal
from io import StringIO
from contextlib import closing
import sys

#TODO: use convective time in compute_wagner_lift
#TODO: truncate action if it would bring the state outside the observation space
#TODO: check_env -> reset() check initialize state[1]

def random_fourier_series(t, T, N):
    A = np.random.normal(0, 1, N)
    #s = A[0]/2*np.ones(len(t))
    #A = np.random.randint(-1, high=2, size=N)
    #s = 0.1*A[0]*np.ones(len(t))
    s = 0.0
    s += 0.01*sum(np.sin(2*math.pi/T*n*t)*A[n]/n for n in range(1, N))
    return s

class WagnerJonesEnv(gym.Env):
    """
    ### Observation Space
    The observation is an `ndarray` with shape `(5 + t_wake_max / delta_t,)` where the elements correspond to the following:
    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | effective angle of attack                                                   | -pi/18 | pi/18  | rad     |
    |   1   | angular velocity of the wing                                                | -pi/18 | pi/18  | rad/s   |
    |   2   | Theodorsen's function state 1 (R.T. Jones approximation)                    | -Inf   | Inf    | -       |
    |   3   | Theodorsen's function state 2 (R.T. Jones approximation)                    | -Inf   | Inf    | -       |

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
        
        self.alpha_eff_threshold = 10 * math.pi / 180
        self.hddot_threshold = np.inf
        self.alpha_dot_threshold = 10 * math.pi / 180
        self.alpha_ddot_threshold = 0.1 #10 * math.pi / 180
        self.lift_threshold = lift_threshold

        self.observe_wake = observe_wake
        self.observe_h_ddot = observe_h_ddot

        # Observations are the wing's effective AOA, the vertical acceleration, the angular velocity and acceleration, and the states for Theodorsen's function approximation. 
        self.state_low = np.array(
            [
                -self.alpha_eff_threshold, # effective AOA
                -self.alpha_dot_threshold, # angular velocity of the wing
                -np.inf,
                -np.inf
            ]
        ).astype(np.float32)

        self.state_high = np.array(
            [
                self.alpha_eff_threshold, # effective AOA
                self.alpha_dot_threshold, # angular velocity of the wing
                np.inf,
                np.inf
            ]
        ).astype(np.float32)
         
        obs_low = self.state_low
        obs_high = self.state_high

        obs_low = self.state_low[0:2]
        obs_high = self.state_high[0:2]

        if self.observe_wake:
            np.append(obs_low, self.state_low[2:])
            np.append(obs_high, self.state_high[2:])
        if self.observe_h_ddot:
            np.append(obs_low, np.float32(self.h_ddot_threshold))
            np.append(obs_high, -np.float32(self.h_ddot_threshold))

        self.observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        # We have 1 action: the angular acceleration
        if self.continuous_actions:
            self.action_space = spaces.Box(-self.alpha_ddot_threshold, self.alpha_ddot_threshold, (1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.discrete_action_values = self.alpha_ddot_threshold * np.linspace(-1, 1, num=self.num_discrete_actions) ** 3

        # Create the discrete system from the continuous system
        # Theodorsen's function:
        C1 = math.pi/2
        C2 = 2*math.pi
        Atilde = np.array([[-0.691, -0.0546], [1, 0]])
        Btilde = np.array([[1], [0]])
        Ctilde = np.array([[0.2161, 0.0273]])
        Dtilde = np.array([[0.5]])
        # Combined pitch and plunge model:
        A = np.block([
            [0, 1, np.zeros((1,2))],
            [0, 0, np.zeros((1,2))],
            [Btilde*C2, np.zeros((2,1)), Atilde], 
        ])
        B = np.array([
            [-1/U, (c/4-a)/U],
            [0, 1],
            [0, 0],
            [0, 0],
        ])
        C = np.block([
            [Dtilde*C2, C1*U, Ctilde]
        ])
        D = np.array([
            [-C1, -C1*a]
        ])
        sys = signal.StateSpace(A, B, C, D)
        sys = sys.to_discrete(delta_t)
        self.A = sys.A.astype(np.float32)
        self.B = sys.B.astype(np.float32)
        self.C = sys.C.astype(np.float32)
        self.D = sys.D.astype(np.float32)

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
            np.append(obs, self.state[2:])
        if self.observe_h_ddot:
            np.append(obs, self.h_ddot)
        return obs

    def _get_info(self):
        return {"previous lift": self.fy, "time": self.t, "time step": self.time_step}


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.time_step = 0

        # Recreate a prescribed vertical acceleration if a random fourier signal is required
        if self.random_fourier_h_ddot:
            self.h_ddot_prescribed = self.U * random_fourier_series(np.linspace(0, self.t_max, int(self.t_max / self.delta_t) + 1), self.h_ddot_T, self.h_ddot_N) 

        # If there is no prescribed vertical acceleration, create a random vertical acceleration
        if self.h_ddot_prescribed is None:
            self.h_ddot = self.np_random.normal(self.h_ddot_mean, self.h_ddot_std)
        else:
            self.h_ddot = self.h_ddot_prescribed[self.time_step]

        self.state = np.zeros(len(self.A), dtype=np.float32)
        self.fy = np.float32(0.0)

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
            alpha_ddot = action[0]
        else:
            alpha_ddot = self.discrete_action_values[action] 

        # Compute the lift and reward
        u = np.array([self.h_ddot, alpha_ddot], dtype=np.float32)
        CL = (np.dot(self.C, self.state) + np.dot(self.D, u))[0]
        self.fy = 0.5 * CL * (self.U ** 2) * self.c * self.rho

        # Compute the reward
        if self.reward_type == 1:
            reward = -(self.fy ** 2) # v1
        elif self.reward_type == 2:
            reward = -(self.fy ** 2 + 0.1 * alpha_ddot ** 2) # v2
        elif self.reward_type == 5:
            reward = -1 * (math.exp((self.fy / self.lift_threshold) ** 2) - 1) + 1 # v5
        elif self.reward_type == 6:
            reward = -10 * (self.fy ** 2) + 1 / (self.t_max / self.delta_t) # v6

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # Update state
        self.state = np.matmul(self.A, self.state) + np.dot(self.B, u)

        # If there is no prescribed vertical acceleration, create a random vertical acceleration
        if self.h_ddot_prescribed is None:
            self.h_ddot = self.np_random.normal(self.h_ddot_mean, self.h_ddot_std)
        else:
            self.h_ddot = self.h_ddot_prescribed[self.time_step]

        # Check if timelimit is reached or state went out of bounds
        #truncated = self.t > self.t_max or math.isclose(self.t, self.t_max, rel_tol=1e-9) or abs(self.fy) > 0.1
        #truncated = self.t > self.t_max or math.isclose(self.t, self.t_max, rel_tol=1e-9)
        if self.t > self.t_max or math.isclose(self.t, self.t_max, rel_tol=1e-9):
            truncated = True
        elif self.state[0] < -self.alpha_eff_threshold or self.state[0] > self.alpha_eff_threshold:
            truncated = True
        elif self.fy < -self.lift_threshold or self.fy > self.lift_threshold:
            #reward -= 10
            truncated = True
        else:
            truncated = False

        # Create observation for next state
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render() 

        return observation, reward, False, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_frame()

    def _render_text(self):
        outfile = StringIO()
        outfile.write("{:5d}{:10.5f} ".format(self.time_step, self.t))
        outfile.write(("{:10.3e} "*3).format(
            self.state[0],
            self.h_ddot,
            self.state[1],
        ))
        with closing(outfile):
            return outfile.getvalue()
