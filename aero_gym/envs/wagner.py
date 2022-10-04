import gym
from gym import spaces
import math
import numpy as np

#TODO: change observation space to one big np.array and combine all states into self.state to avoid having to cast everything to float32

def compute_wagner_lift(h_dot, h_ddot, alpha, Omega, Omega_dot, wake_circulation, t, delta_t, rho=1, U=1, c=1, a=0):
    t_release_earliest_wake_element = (t / delta_t - len(wake_circulation) + 1) * delta_t
    fy_wake = wake_circulation[-1] * wagner(t_release_earliest_wake_element)
    for i in range(1,len(wake_circulation)):
        fy_wake += (wake_circulation[-i - 1] - wake_circulation[-i]) * wagner(t_release_earliest_wake_element - i * delta_t)
    fy_wake *= rho * U
    fy = rho * c ** 2 * math.pi / 4 * (-h_ddot + a * Omega_dot - U * Omega) + fy_wake
    return fy

def compute_Gamma_b(h_dot, alpha, Omega, U=1, c=1, a=0):
    return math.pi * c * (h_dot + U * alpha + (c / 4 - a) * Omega) 

def compute_Gamma_b_dot(h_ddot, Omega, Omega_dot, U=1, c=1, a=0):
    return math.pi * c * (h_ddot + U * Omega + (c / 4 - a) * Omega_dot) 

def wagner(t):
    return 1 - 0.165 * math.exp(-0.091 * t) - 0.335 * math.exp(-0.6 * t)

class WagnerEnv(gym.Env):
    """
    ### Observation Space
    The observation is an `ndarray` with shape `(5 + t_wake_max / delta_t,)` where the elements correspond to the following:
    | Index | Observation                                                                 | Min   | Max  | Unit    |
    |-------|-----------------------------------------------------------------------------|-------|------|---------|
    |   0   | vertical velocity of the wing                                               | -Inf  | Inf  | m/s     |
    |   1   | vertical acceleration of the wing                                           | -Inf  | Inf  | m/s^2   |
    |   2   | angle of the wing (pos counterclockwise)                                    | -pi/2 | pi/2 | rad     |
    |   3   | angular velocity of the wing                                                | -Inf  | Inf  | rad/s   |
    |   4   | angular acceleration of the wing                                            | -Inf  | Inf  | rad/s^2 |
    |   5   | local circulation of the wake                                               | -Inf  | Inf  | m/s     |
    |   6   | local circulation of the wake minus the part released at t                  | -Inf  | Inf  | m/s     |
    |   7   | local circulation of the wake minus the parts released at t and t - delta_t | -Inf  | Inf  | m/s     |
    |   .   |                                     .                                       |   .   |   .  |    .    | 
    |   .   |                                     .                                       |   .   |   .  |    .    | 
    |   .   |                                     .                                       |   .   |   .  |    .    | 
    |  -1   | local circulation of the wake part released at t - t_wake_max               | -Inf  | Inf  | m/s     |

    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, delta_t=0.1, t_max=100.0, t_wake_max=20.0 ,h_ddot_mean=0.0, h_ddot_std=1.0, h_ddot_prescribed=None, steady_ics=True, zero_ics=True):
        self.delta_t = delta_t  # The time step size of the simulation
        self.t_max = t_max
        self.t_wake_max = t_wake_max
        self.N_wake = int(self.t_wake_max / self.delta_t)
        self.steady_ics = steady_ics
        self.zero_ics = zero_ics
        if h_ddot_prescribed is not None:
            assert len(h_ddot_prescribed) > t_max / delta_t, "The prescribed vertical acceleration has not enough entries for the whole simulation"
        self.h_ddot_prescribed = h_ddot_prescribed

        # Observations are the wing's AOA, the angular/vertical velocity and acceleration, and the circulation of the elements in the wake. If we wouldn't include the state of the wake, it wouldn't be an MDP.
        low = np.concatenate(
            np.array(
                [
                    -np.inf,
                    -np.inf,
                    -math.pi/2,
                    -np.inf,
                    -np.inf,
                ],
            np.full((self.N_wake,), -np.inf)
        ).astype(np.float32)

        high = np.concatenate(
            np.array(
                [
                    np.inf,
                    np.inf,
                    math.pi/2,
                    np.inf,
                    np.inf,
                ],
            np.full((self.N_wake,), np.inf)
        ).astype(np.float32)

        self.observation_space = spaces.Box(low, high, (5 + self.N_wake,), dtype=np.float32)

        # We have 1 action: the angular acceleration
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32)

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
        return self.state

    def _get_info(self):
        return {"lift": self.fy}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.step = 0
       
        if self.zero_ics:
            self.state = np.zeros(5 + self.N_wake, dtype=np.float32)
        elif self.steady_ics:
            self.state = np.concatenate(
                np.array(
                    [
                        self.np_random.uniform(-1.0, 1.0),
                        0.0,
                        self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180),
                        0.0,
                        0.0,
                    ]
                ).astype(np.float32),
                np.zeros(self.N_wake, dtype=np.float32)
            )
        else:
            self.state = np.concatenate(
                np.array(
                    [
                        self.np_random.uniform(-1.0, 1.0),
                        self.np_random.uniform(-1.0, 1.0),
                        self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180),
                        self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180),
                        self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180),
                    ]
                ).astype(np.float32),
                self.np_random.uniform(-1.0, 1.0, self.N_wake, dtype=np.float32).astype(np.float32)
            )
            # Set the last released wake element such that it is compatible with the latest change in 
            self.state[5] = self.state[6] - self.delta_t * compute_Gamma_b_dot(self.state[1], self.state[3], self.state[4])
        
        # Compute the lift
        self.fy = compute_wagner_lift(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5:-1], self.t, self.delta_t)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Update AOA
        self.state[4] = action
        self.state[3] += self.state[4] * self.delta_t
        self.state[2] += self.state[3] * self.delta_t
        
        # If there is no prescribed vertical acceleration, sample the vertical acceleration from a normal distribution and update the vertical velocity
        if h_ddot_prescribed is None:
            self.h_ddot = self.np_random.normal(self.h_ddot_mean, self.h_ddot_std)
        else:
            self.h_ddot = h_ddot_prescribed[self.step]
        self.state[0] += self.h_ddot * self.delta_t

        # Update wake
        self.state[6:-1] = self.state[5:-2]
        self.state[5] = self.state[6] - self.delta_t * compute_Gamma_b_dot(self.state[1], self.state[3], self.state[4])

        # Compute the lift and reward
        self.fy = compute_wagner_lift(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5:-1], self.t, self.delta_t)
        reward = -abs(self.fy)
        
        # Update the time and time step
        self.t += delta_t
        self.step += 1

        # Check if the episode is done
        terminated = self.t >= self.tmax

        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame() 

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_frame()

    def _render_text(self):
        outfile = StringIO()
        outfile.write("test\n")
        with closing(outfile):
            return outfile.getvalue()
