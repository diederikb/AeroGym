import gym
from gym import spaces
import math
import numpy as np

def compute_wagner_lift(alpha, Omega, Omega_dot, h_dot, h_ddot, wake_circulation, t, delta_t, rho=1, U=1, c=1, a=0):
    t_release_earliest_wake_element = (t / delta_t - len(wake_circulation) + 1) * delta_t
    fy_wake = wake_circulation[-1] * wagner(t_release_earliest_wake_element)
    for i in range(1,len(wake_circulation)):
        fy_wake += (wake_circulation[-i - 1] - wake_circulation[-i]) * wagner(t_release_earliest_wake_element - i * delta_t)
    fy_wake *= -rho * U
    fy = rho * c ** 2 * math.pi / 4 * (-h_ddot + a * Omega_dot - U * Omega) + fy_wake
    return fy

def compute_Gamma_b(alpha, Omega, h_dot, U=1, c=1, a=0):
    return math.pi * c * (h_dot + U * alpha + (c / 4 - a) * Omega) 

def compute_Gamma_b_dot(Omega, Omega_dot, h_ddot, U=1, c=1, a=0):
    return math.pi * c * (h_ddot + U * Omega + (c / 4 - a) * Omega_dot) 

def wagner(t):
    return 1 - 0.165 * math.exp(-0.091 * t) - 0.335 * math.exp(-0.6 * t)

class WagnerEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, delta_t=0.1, t_max=100.0, t_wake_max=20.0 ,h_ddot_mean=0.0, h_ddot_std=1.0, h_ddot_prescribed=None, steady_ics=True, zero_ics=True):
        self.delta_t = delta_t  # The time step size of the simulation
        self.t_max = t_max
        self.t_wake_max = t_wake_max
        self.steady_ics = steady_ics
        self.zero_ics = zero_ics
        if h_ddot_prescribed is not None:
            assert len(h_ddot_prescribed) > t_max / delta_t, "The prescribed vertical acceleration has not enough entries for the whole simulation"
        self.h_ddot_prescribed = h_ddot_prescribed

        # Observations are the wing's AOA, the angular/vertical velocity and acceleration, and the circulation of the elements in the wake. If we wouldn't include the state of the wake, it wouldn't be an MDP.
        self.observation_space = spaces.Dict(
            {
                "alpha": spaces.Box(5*math.pi/180, 5*math.pi/180, dtype=np.float32),
                "Omega": spaces.Box(5*math.pi/180, 5*math.pi/180, dtype=np.float32),
                "Omega_dot": spaces.Box(5*math.pi/180, 5*math.pi/180, dtype=np.float32),
                "h_dot": spaces.Box(-1, 1, dtype=float32),
                "h_ddot": spaces.Box(-1, 1, dtype=float32),
                "wake_circulation": spaces.Box(low=-1, high=1, shape=(t_wake_max / delta_t,), dtype=np.float32),
            }
        )

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
        return {"alpha":self._alpha, "Omega": self._Omega, "Omega_dot": self._Omega_dot, "h_dot": self._h_dot, "h_ddot": self._h_ddot, "wake_circulation": self._wake_circulation}

    def _get_info(self):
        return {self.fy}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.step = 0
       
        if self.zero_ics:
            self._alpha = 0.0
            self._Omega = 0.0
            self._Omega_dot = 0.0
            self._h_dot = 0.0
            self._h_ddot = 0.0 
            self._wake_circulation = np.zeros(self.t_wake_max / self.delta_t)
        elif self.steady_ics:
            self._alpha = self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180)
            self._Omega = 0.0
            self._Omega_dot = 0.0
            self._h_dot = self.np_random.uniform(-1.0, 1.0)
            self._h_ddot = 0.0 
            self._wake_circulation = np.zeros(self.t_wake_max / self.delta_t)
        else:
            self._alpha = self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180)
            self._Omega = self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180)
            self._Omega_dot = self.np_random.uniform(-5.0*math.pi/180, 5.0*math.pi/180)
            self._h_dot = self.np_random.uniform(-1.0, 1.0)
            self._h_ddot = self.np_random.uniform(-1.0, 1.0)
            self._wake_circulation = self.np_random.uniform(-1.0, 1.0, self.t_wake_max / self.delta_t)
            self._wake_circulation[0] = self._wake_circulation[1] + self.delta_t * compute_Gamma_b_dot(self._Omega, self._Omega_dot, self._h_ddot)
        
        # Compute the lift
        self.fy = compute_wagner_lift(self._alpha, self._Omega, self._Omega_dot, self._h_dot, self._h_ddot, self._wake_circulation, self.t, self.delta_t)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Update AOA
        self._Omega_dot = action
        self._Omega += self._Omega_dot * self._delta_t
        self._alpha += self._Omega * self.delta_t
        
        # If there is no prescribed vertical acceleration, sample the vertical acceleration from a normal distribution and update the vertical velocity
        if h_ddot_prescribed is None:
            self.h_ddot = self.np_random.normal(self.h_ddot_mean, self.h_ddot_std)
        else:
            self.h_ddot = h_ddot_prescribed[self.step]
        self._h_dot += self.h_ddot * self.delta_t

        # Update wake
        self._wake_circulation = np.roll(self._wake_circulation, 1)
        self._wake_circulation[0] = self._wake_circulation[1] + self.delta_t * compute_Gamma_b_dot(self._Omega, self._Omega_dot, self._h_ddot)

        # Compute the lift and reward
        self.fy = compute_wagner_lift(self._alpha, self._Omega, self._Omega_dot, self._h_dot, self._h_ddot, self._wake_circulation, self.delta_t, self.t)
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
