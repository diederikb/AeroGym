import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy import signal
from io import StringIO
from contextlib import closing

#TODO: use observation wrappers (with dicts) instead of observe_wake and observe_h_ddot

def compute_alpha(h_dot, alpha_eff, alpha_dot, U=1.0, c=1.0, a=1.0):
    """
    Compute the angle of attack from the effective angle of attack, angular velocity, and vertical velocity.
    """
    alpha = h_dot / U + alpha_eff - (0.25 * c - a) * alpha_dot / U
    return alpha

class WagnerJonesEnv(gym.Env):
    """
    ### Observation Space

    The default observation space is an `ndarray` with shape `(4,)` where the elements correspond to the following:
    The observed angle of attack is by default the angle of the wing with U but can be changed to the effective one by setting `observed_alpha_is_eff`.
    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | (effective) angle of attack at the current timestep                         | -pi/18 | pi/18  | rad     |
    |   1   | angular velocity of the wing at the current timestep                        | -pi/18 | pi/18  | rad/s   |
    Setting the following keyword arguments to `True` will append the observation space with the following arrays (in the order that is given here):

    `observe_wake` (N = `t_max` / `delta_t`)

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | R.T. Jones approximation function state 1 at the current timestep           | -Inf   | Inf    | -       |
    |   1   | R.T. Jones approximation function state 2 at the current timestep           | -Inf   | Inf    | -       |
    `observe_h_ddot`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical acceleration of the wing at the current timestep                   |-0.1U/dt| 0.1U/dt| m/s^2   |

    `observe_previous_lift`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | lift at the previous timestep (per unit depth)                              |see args|see args| kg/s^2  |
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
                 observed_alpha_is_eff=False,
                 observe_wake=False,
                 observe_h_ddot=False,
                 observe_previous_lift=False,
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

        self.h_dot_threshold = 0.1 * U
        self.h_ddot_threshold = 0.1 * U / delta_t
        self.alpha_eff_threshold = 10 * np.pi / 180
        self.alpha_dot_threshold = 10 * np.pi / 180
        self.alpha_ddot_threshold = alpha_ddot_threshold
        self.lift_threshold = lift_threshold

        self.observed_alpha_is_eff = observed_alpha_is_eff
        self.observe_wake = observe_wake
        self.observe_h_ddot = observe_h_ddot
        self.observe_previous_lift = observe_previous_lift

        # The observations don't include the vertical velocity
        # The first element is either the effective AOA or the actual one
        obs_low = np.array(
            [
                -self.alpha_eff_threshold, # effective AOA
                -self.alpha_dot_threshold, # angular velocity of the wing
            ]
        )

        obs_high = np.array(
            [
                self.alpha_eff_threshold, # effective AOA
                self.alpha_dot_threshold, # angular velocity of the wing
            ]
        )

        if self.observe_wake:
            obs_low = np.append(obs_low, np.full(2, -np.inf))
            obs_high = np.append(obs_high, np.full(2, np.inf))
        if self.observe_h_ddot:
            obs_low = np.append(obs_low, -self.h_ddot_threshold)
            obs_high = np.append(obs_high, self.h_ddot_threshold)
        if self.observe_previous_lift:
            obs_low = np.append(obs_low, -self.lift_threshold)
            obs_high = np.append(obs_high, self.lift_threshold)

        self.observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        # We have 1 action: the angular acceleration
        if self.continuous_actions:
            # Will be rescaled by the threshold
            self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.discrete_action_values = self.alpha_ddot_threshold * np.linspace(-1, 1, num=self.num_discrete_actions) ** 3

        # Create the discrete system from the continuous system
        # Theodorsen's function:
        C1 = np.pi/2
        C2 = 2*np.pi
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
            [Dtilde*C2, C1*c/U, Ctilde]
        ])
        # Only added mass
        #C = np.block([
        #    [0, C1*U, np.zeros((1,2))]
        #])
        D = np.array([
            [-C1*c/U**2, -C1*a*c/U**2]
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
        if not self.observed_alpha_is_eff:
            alpha = compute_alpha(self.h_dot, self.state[0], self.state[1], U=self.U, c=self.c, a=self.a)
            obs = np.array([alpha, self.state[1]])
        else:
            obs = self.state[0:2]
        if self.observe_wake:
            obs = np.append(obs, self.state[2:])
        if self.observe_h_ddot:
            obs = np.append(obs, self.h_ddot)
        if self.observe_previous_lift:
            obs = np.append(obs, self.fy)
        return obs.astype(np.float32)

    def _get_info(self):
        return {"previous fy": self.fy,
                "previous alpha_ddot": self.alpha_ddot,
                "current alpha_dot": self.state[1],
                "current alpha_eff": self.state[0],
                "current h_ddot": self.h_ddot,
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

        self.h_dot = 0.0 # Keep track of h_dot to compute alpha from alpha_eff
        self.h_ddot = self.h_ddot_list[self.time_step]
        self.state = np.zeros(len(self.A))
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
        u = np.array([self.h_ddot, self.alpha_ddot], dtype=np.float32)
        CL = (np.dot(self.C, self.state) + np.dot(self.D, u))[0]
        self.fy = 0.5 * CL * (self.U ** 2) * self.c * self.rho

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
        else:
            raise NotImplementedError("Specified reward type is not implemented.")

        # Update the time and time step
        self.t += self.delta_t
        self.time_step += 1

        # Update state
        self.state = np.matmul(self.A, self.state) + np.dot(self.B, u)
        self.h_dot = self.h_dot + self.delta_t * self.h_ddot

        # Check if timelimit is reached
        if self.t > self.t_max or np.isclose(self.t, self.t_max, rtol=1e-9):
            self.truncated = True
        else:
            self.h_ddot = self.h_ddot_list[self.time_step]

        # Check if alpha_eff or lift goes out of bounds
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
        if self.observed_alpha_is_eff:
            shown_alpha = self.state[0]
        else:
            shown_alpha = compute_alpha(self.h_dot, self.state[0], self.state[1], U=self.U, c=self.c, a=self.a)
        outfile = StringIO()
        outfile.write("{:5d}{:10.5f}".format(self.time_step, self.t))
        outfile.write((" {:10.3e}" * 4).format(
            self.h_dot,
            shown_alpha,
            self.state[1],
            self.h_ddot,
        ))
        with closing(outfile):
            return outfile.getvalue()
