import gym
from gym import spaces
import math
import numpy as np
from io import StringIO
from contextlib import closing

#TODO: use convective time in compute_wagner_lift
#TODO: truncate action if it would bring the state outside the observation space
#TODO: check_env -> reset() check initialize state[1]

def compute_wagner_lift(h_dot, h_ddot, alpha, Omega, Omega_dot, wake_circulation, t, delta_t, rho=1, U=1, c=1, a=0):
    t_after_release_earliest_wake_element = len(wake_circulation) * delta_t
    fy_wake = wake_circulation[-1] * wagner(t_after_release_earliest_wake_element)
    for i in range(1,len(wake_circulation)):
        fy_wake += (wake_circulation[-i - 1] - wake_circulation[-i]) * wagner(t_after_release_earliest_wake_element - i * delta_t)
    fy_wake *= rho * U
    fy_wake = 0
    fy = rho * c ** 2 * math.pi / 4 * (-h_ddot + a * Omega_dot - U * Omega) + fy_wake
    return fy

def compute_Gamma_b(h_dot, alpha, Omega, U=1, c=1, a=0):
    return math.pi * c * (h_dot + U * alpha + (c / 4 - a) * Omega) 

def compute_Gamma_b_dot(h_ddot, Omega, Omega_dot, U=1, c=1, a=0):
    return math.pi * c * (h_ddot + U * Omega + (c / 4 - a) * Omega_dot) 

def wagner(t):
    if t >= 0.0:
        return 1.0 - 0.165 * math.exp(-0.091 * t) - 0.335 * math.exp(-0.6 * t)
    else:
        return 0.0

def random_fourier_series(t, T, N):
    #A = np.random.normal(0, 1, N)
    #s = A[0]/2*np.ones(len(t))
    A = np.random.randint(-1, high=2, size=N)
    s = 0.1*A[0]*np.ones(len(t))
    s += sum(np.sin(2*math.pi/T*n*t)*A[n]/n for n in range(1, N))
    return s

class WagnerEnv(gym.Env):
    """
    ### Observation Space
    The observation is an `ndarray` with shape `(5 + t_wake_max / delta_t,)` where the elements correspond to the following:
    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical velocity of the wing                                               | -0.1*U | 0.1*U  | m/s     |
    |   1   | vertical acceleration of the wing                                           | -Inf   | Inf    | m/s^2   |
    |   2   | angle of the wing (pos counterclockwise)                                    | -pi/36 | pi/36  | rad     |
    |   3   | angular velocity of the wing                                                | -Inf   | Inf    | rad/s   |
    |   4   | angular acceleration of the wing                                            | -Inf   | Inf    | rad/s^2 |
    |   5   | local circulation of the wake                                               | -Inf   | Inf    | m/s     |
    |   6   | local circulation of the wake minus the part released at t                  | -Inf   | Inf    | m/s     |
    |   7   | local circulation of the wake minus the parts released at t and t - delta_t | -Inf   | Inf    | m/s     |
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |   .   |                                     .                                       |   .    |  .     |  .      | 
    |  -1   | local circulation of the wake part released at t - t_wake_max               | -Inf   | Inf    | m/s     |

    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, delta_t=0.1, t_max=100.0, t_wake_max=20.0, h_ddot_mean=0.0, h_ddot_std=1.0, h_ddot_N=100, rho=1.0, U=1.0, c=1, a=0, h_ddot_prescribed=None, random_ics=False, random_fourier_h_ddot=False):
        self.state = None
        self.delta_t = delta_t  # The time step size of the simulation
        self.t_max = t_max
        self.t_wake_max = t_wake_max
        self.N_wake = int(self.t_wake_max / self.delta_t)
        if h_ddot_prescribed is not None:
            assert len(h_ddot_prescribed) >= int(t_max / delta_t) + 1, "The prescribed vertical acceleration has not enough entries for the whole simulation (including t=0)"
        self.h_ddot_prescribed = h_ddot_prescribed
        self.h_ddot_mean = h_ddot_mean
        self.h_ddot_std = h_ddot_std
        self.h_ddot_N = h_ddot_N
        self.rho = rho
        self.U = U
        self.c = c
        self.a = a
        self.random_fourier_h_ddot = random_fourier_h_ddot
        self.random_ics = random_ics

        # Observations are the wing's AOA, the angular/vertical velocity and acceleration, and the circulation of the elements in the wake. If we wouldn't include the state of the wake, it wouldn't be an MDP.
        #low = np.concatenate(
        #    (
        #        np.array(
        #            [
        #                -0.1 * U,
        #                -np.inf,
        #                -math.pi/36,
        #                -np.inf,
        #                -np.inf,
        #            ]
        #        ),
        #        np.full((self.N_wake,), -np.inf)
        #    )
        #).astype(np.float32)

        #high = np.concatenate(
        #    (
        #        np.array(
        #            [
        #                0.1 * U,
        #                np.inf,
        #                math.pi/36,
        #                np.inf,
        #                np.inf,
        #            ]
        #        ),
        #        np.full((self.N_wake,), np.inf)
        #    )
        #).astype(np.float32)
        low = np.concatenate(
            (
                np.array(
                    [
                        -0.1 * U,
                        -np.inf,
                        -math.pi/36,
                        -np.inf,
                        -np.inf,
                    ]
                ),
            )
        ).astype(np.float32)

        high = np.concatenate(
            (
                np.array(
                    [
                        0.1 * U,
                        np.inf,
                        math.pi/36,
                        np.inf,
                        np.inf,
                    ]
                ),
            )
        ).astype(np.float32)

        #self.observation_space = spaces.Box(low, high, (5 + self.N_wake,), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, (5,), dtype=np.float32)

        # We have 1 action: the angular acceleration
        #self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

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
        return self.state[0:5]

    def _get_info(self):
        return {"lift": self.fy, "time": self.t, "time step": self.time_step}


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.time_step = 0

        if self.random_ics:
            self.state = np.concatenate(
                (
                    np.array(
                        [
                            self.np_random.uniform(-1.0, 1.0),
                            self.np_random.uniform(-1.0, 1.0),
                            self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180),
                            self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180),
                            self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180),
                        ]
                    ).astype(np.float32),
                    self.np_random.uniform(-1.0, 1.0, self.N_wake).astype(np.float32)
                )
            )
            # Set the last released wake element such that it is compatible with the latest change in 
            self.state[5] = self.state[6] - self.delta_t * compute_Gamma_b_dot(self.state[1], self.state[3], self.state[4], U=self.U, c=self.c, a=self.a)
        else:
            self.state = np.zeros(5 + self.N_wake, dtype=np.float32)
        
        # Recreate a prescribed vertical acceleration if a random fourier signal is required
        if self.random_fourier_h_ddot:
            self.h_ddot_prescribed = self.U * random_fourier_series(np.linspace(0, self.t_max, int(self.t_max / self.delta_t) + 1), self.t_max, self.h_ddot_N) 
            self.state[1] = self.h_ddot_prescribed[0]

        
        # Compute the lift
        self.fy = compute_wagner_lift(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5:-1], self.t, self.delta_t, rho=self.rho, U=self.U, c=self.c, a=self.a)

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render() 

        return observation

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Cannot call env.step() before calling reset()"

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # If there is no prescribed vertical acceleration, create a random vertical acceleration
        if self.h_ddot_prescribed is None:
            h_ddot_np1 = self.np_random.normal(self.h_ddot_mean, self.h_ddot_std)
        else:
            h_ddot_np1 = self.h_ddot_prescribed[self.time_step]
        # Update vertical velocity and acceleration
        self.state[0] += 0.5 * (self.state[1] + h_ddot_np1) * self.delta_t
        self.state[1] = h_ddot_np1

        # Update AOA
        if action == 1:
            Omega_dot_np1 = 0.1
        elif action == 2:
            Omega_dot_np1 = -0.1
        else:
            Omega_dot_np1 = 0

        Omega_np1 = self.state[3] + 0.5 * (self.state[4] + Omega_dot_np1) * self.delta_t
        alpha_np1 = self.state[2] + 0.5 * (self.state[3] + Omega_np1) * self.delta_t
        self.state[2] = alpha_np1
        self.state[3] = Omega_np1
        self.state[4] = Omega_dot_np1

        # Update wake
        self.state[6:] = self.state[5:-1]
        self.state[5] = -compute_Gamma_b(self.state[0], self.state[2], self.state[3], U=self.U, c=self.c, a=self.a)
        #self.state[5] = self.state[6] - self.delta_t * compute_Gamma_b_dot(self.state[1], self.state[3], self.state[4], U=self.U)

        # Compute the lift and reward
        self.fy = compute_wagner_lift(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5:-1], self.t, self.delta_t, rho=self.rho, U=self.U, c=self.c, a=self.a)
        #reward = -(self.fy ** 2 + 0.01 * self.state[4] ** 2)
        #reward = -(self.state[4] ** 2)
        if abs(self.fy) > 0.1:
            reward = -1 
        else:
            reward = -(self.fy ** 2) - 0.1 * (self.state[4] ** 2)
        # Check if the episode is done
        truncated = self.t >= self.t_max
        terminated = abs(self.fy) > 0.1
        done = bool(truncated or terminated)

        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render() 

        return observation, reward, done, info

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self._render_text()
        else:
            return self._render_frame()

    def _render_text(self):
        outfile = StringIO()
        outfile.write("{:5d}{:10.5f} ".format(self.time_step, self.t))
        outfile.write((" {:10.3e}").format(self.fy))
        outfile.write((" {:10.3e}"*5).format(*self.state[0:5]))
        outfile.write("\n")
        with closing(outfile):
            return outfile.getvalue()
