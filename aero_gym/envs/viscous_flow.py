import gymnasium as gym
from gymnasium import spaces
import numpy as np
from io import StringIO
from contextlib import closing
from julia import Julia
import importlib.resources
import os

# TODO: compute fy in step() in a better way

class ViscousFlowEnv(gym.Env):
    """
    ## Description

    The ViscousFlow environment is the two-dimensional, viscous, aerodynamic model for an airfoil undergoing arbitrary motions. The airfoil undergoes prescribed or random vertical accelerations and the goal is to minimize the lift variations by controlling the AOA through the angular acceleration of the airfoil.

    TODO

    ## Action space

    The action is a `numpy.ndarray(dtype=numpy.float32)` with shape `(1,)` which can take values in the range `[-1,1]` and represents the scaled angular acceleration (alpha_ddot) applied at a distance `a` from the midchord position.
    The actual angular acceleration depends on the value of the argument `alpha_ddot_scale`: `alpha_ddot = action * alpha_ddot_scale`.

    Alternatively, if `use_discrete_actions` is `True`, the action is a `numpy.int64` which can take discrete values `{0, 1, ..., N-1}`, with `N = `num_discrete_actions`, in which case the action `i` gets mapped to the angular acceleration `(2 * i  / ( N - 1 ) - 1 ) * alpha_ddot_scale`

    ## Observation Space

    TODO: explain scaling

    The default observation space is an `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | angle of attack at the current timestep                                     | -Inf   |  Inf   | rad     |
    |   1   | angular velocity of the wing at the current timestep                        | -Inf   |  Inf   | rad/T   |

    Setting the following keyword arguments to `True` will append the observation space with the following arrays (in the order that is given here):

    `observe_vorticity_field`
    `observe_vorticity_field`

    `observe_h_ddot`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical acceleration of the wing at the current timestep                   | -Inf   |  Inf   | L/T^2   |

    `observe_h_dot`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | vertical velocity of the wing at the current timestep                       | -Inf   |  Inf   | L/T     |
    
    `observe_previous_lift`

    | Index | Observation                                                                 | Min    | Max    | Unit    |
    |-------|-----------------------------------------------------------------------------|--------|--------|---------|
    |   0   | lift at the previous timestep (per unit depth)                              | -Inf   |  Inf   | M/T^2   |

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

    The episode time, kinematic states, flow states, and previous lift are all initialized to zero. The vertical acceleration is initialized to its first value.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination: Absolute value of the lift is greater than `lift_scale`
    2. Truncation: Episode time `t` is greater than or equal to `t_max`

    ## Arguments

    TODO

    """
    metadata = {"render_modes": ["ansi", "grayscale_array"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 initialization_time=5.0,
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
                 xlim=[-0.75,2.0],
                 ylim=[-0.5,0.5],
                 reward_type=3,
                 observe_vorticity_field=False,
                 normalize_vorticity=True,
                 observe_h_ddot=False,
                 observe_h_dot=False,
                 observe_previous_lift=False,
                 observe_previous_lift_error=False,
                 observe_previous_pressure=False,
                 pressure_sensor_positions=[],
                 lift_termination=False,
                 lift_upper_limit=None,
                 lift_lower_limit=None,
                 h_dot_termination=True,
                 alpha_dot_termination=True,
                 alpha_termination=True,
                 lift_scale=1.0,
                 alpha_ddot_scale=1.0,
                 alpha_dot_scale=1.0,
                 alpha_scale=1.0,
                 h_ddot_scale=1.0,
                 h_dot_scale=1.0,
                 vorticity_scale=1.0):
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
        self.reward_type = reward_type

        self.h_dot_scale = h_dot_scale
        self.h_ddot_scale = h_ddot_scale
        self.alpha_eff_scale = lift_scale
        self.alpha_scale = alpha_scale
        self.alpha_dot_scale = alpha_dot_scale
        self.alpha_ddot_scale = alpha_ddot_scale
        self.lift_scale = lift_scale
        self.vorticity_scale = vorticity_scale
        self.pressure_scale = lift_scale / c
        self.lift_termination = lift_termination
        self.h_dot_termination = h_dot_termination
        self.alpha_dot_termination = alpha_dot_termination
        self.alpha_termination = alpha_termination
        self.h_dot_limit = U
        self.alpha_dot_limit = U / (c / 2 + np.abs(a))
        self.alpha_limit = 80 * np.pi / 180

        self.observe_vorticity_field = observe_vorticity_field
        self.normalize_vorticity = normalize_vorticity
        self.observe_h_ddot = observe_h_ddot
        self.observe_h_dot = observe_h_dot
        self.observe_previous_lift = observe_previous_lift
        self.observe_previous_lift_error = observe_previous_lift_error
        self.observe_previous_pressure = observe_previous_pressure
        self.pressure_sensor_positions = np.array(pressure_sensor_positions)

        # Set upper and lower lift limits to lift scale if they were not provided
        if lift_upper_limit is None or lift_lower_limit is None:
            print("`lift_upper_limit` and/or `lift_lower_limit` not provided, setting limits to plus and minus `lift_scale`")
            self.lift_upper_limit = self.lift_scale
            self.lift_lower_limit = -self.lift_scale
        else:
            self.lift_upper_limit = lift_upper_limit
            self.lift_lower_limit = lift_lower_limit

        # For testing:
        Re = 200.0
        grid_Re = 4.0

        # Create the Julia process and set up the viscous flow simulation
        self.jl = Julia()
        julia_sys_setup_commands_template = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_sys_setup_commands.txt").read_text()
        julia_sys_setup_commands = julia_sys_setup_commands_template.format(
                Re=Re,
                grid_Re=grid_Re,
                xmin=xlim[0],
                xmax=xlim[1],
                ymin=ylim[0],
                ymax=ylim[1],
                U=U,
                c=c,
                a=a,
                init_time=initialization_time)
        self.jl.eval(julia_sys_setup_commands)

        # Get the timestep from julia
        delta_t_solver = self.jl.eval("sys.timestep_func(u0, sys)")
        # For now, make sure that the env delta_t is a multiple of delta_t_solver
        self.n_solver_steps_per_env_step = int(np.floor(delta_t / delta_t_solver))
        self.delta_t = self.n_solver_steps_per_env_step * delta_t_solver
        print("setting timestep to multiple of flow solver timestep: delta_t = " + str(self.delta_t))
        
        # The default observation
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
        
        # Extra observations specified with keyword arguments
        if self.observe_h_ddot:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_lift_error:
            obs_low = np.append(obs_low, -np.inf)
            obs_high = np.append(obs_high, np.inf)
        if self.observe_previous_pressure:
            obs_low = np.append(obs_low, np.full_like(self.pressure_sensor_positions, -np.inf))
            obs_high = np.append(obs_high, np.full_like(self.pressure_sensor_positions, np.inf))

        scalar_observation_space = spaces.Box(obs_low, obs_high, (len(obs_low),), dtype=np.float32)

        if self.observe_vorticity_field:
            nx, ny = self.jl.eval("size(zeros_gridcurl(sys))")
            if self.normalize_vorticity: # treat field as a grayscale image
                vorticity_observation_space = spaces.Box(0, 255, shape=(ny, nx, 1), dtype=np.uint8)
            else:
                vorticity_observation_space = spaces.Box(np.inf, -np.inf, shape=(ny, nx, 1), dtype=np.float32)
            self.observation_space = spaces.Dict({"vorticity": vorticity_observation_space, "scalars": scalar_observation_space})
        else:
            self.observation_space = scalar_observation_space

        # We have 1 action: the angular acceleration
        if self.use_discrete_actions:
            self.action_space = spaces.Discrete(self.num_discrete_actions)
        else:
            # Will be rescaled by the threshold
            self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.discrete_action_values = self.alpha_ddot_scale * np.linspace(-1, 1, num=self.num_discrete_actions) ** 3

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
        # get the kinematic state from julia
        pos = self.jl.eval("exogenous_position_vector(aux_state(integrator.u),m,1)")
        vel = self.jl.eval("exogenous_velocity_vector(aux_state(integrator.u),m,1)")
        scalar_obs = np.array([-pos[0] / self.alpha_scale, -vel[0] / self.alpha_dot_scale])

        # create observation vector
        if self.observe_h_ddot:
            scalar_obs = np.append(scalar_obs, self.h_ddot / self.h_ddot_scale)
        if self.observe_h_dot:
            scalar_obs = np.append(scalar_obs, vel[1] / self.h_dot_scale)
        if self.observe_previous_lift:
            scalar_obs = np.append(scalar_obs, self.fy / self.lift_scale)
        if self.observe_previous_lift_error:
            scalar_obs = np.append(scalar_obs, self.fy_error / self.lift_scale)
        if self.observe_previous_pressure:
            p_body = self.jl.eval("pressurejump(integrator).data")
            x_body = self.jl.eval("collect(body)[1]")
            self.p = np.interp(self.pressure_sensor_positions, x_body, p_body)
            scaled_pressure = [p / self.pressure_scale for p in self.p]
            scalar_obs = np.append(scalar_obs, scaled_pressure)

        scalar_obs = scalar_obs.astype(np.float32) 

        if self.observe_vorticity_field:
            vorticity_field = np.flip(np.transpose(self.jl.eval("vorticity(integrator).data")),axis=0)
            if self.normalize_vorticity:
                # quick and dirty mapping from signed float to unsigned int
                vorticity_obs = np.round(np.clip(vorticity_field / (2 * self.vorticity_scale) + 0.5, 0, 1) * 255).astype(np.uint8)
                # add extra dim that indicates the image as a single channel (check if this adds extra memory and if it should be made more efficient)
                vorticity_obs = np.expand_dims(vorticity_obs, axis=2)
            else:
                vorticity_obs = vorticity_field / self.vorticity_scale
            obs = {"vorticity": vorticity_obs, "scalars": scalar_obs}
        else:
            obs = scalar_obs

        return obs

    def _get_info(self):
        pos = self.jl.eval("exogenous_position_vector(aux_state(integrator.u),m,1)")
        vel = self.jl.eval("exogenous_velocity_vector(aux_state(integrator.u),m,1)")
        return {"previous scaled fy": self.fy / self.lift_scale,
                "previous scaled fy_error": self.fy_error / self.lift_scale,
                "previous scaled alpha_ddot": self.alpha_ddot / self.alpha_ddot_scale,
                "scaled alpha_dot": -vel[0] / self.alpha_dot_scale,
                "scaled alpha": -pos[0] / self.alpha_scale,
                "scaled h_ddot": self.h_ddot / self.h_ddot_scale,
                "scaled h_dot": vel[1] / self.h_dot_scale,
                "scaled reference_lift": self.reference_lift / self.lift_scale,
                "t": self.t,
                "time_step": self.time_step}

    def reset(self, seed=None, options=None):
        os.system("vmstat")
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the time and time step
        self.t = 0.0
        self.time_step = 0
        self.truncated = False
        self.terminated = False

        # If there is no prescribed vertical acceleration use the provided function to generate one. If no function was provided, set the vertical acceleration to zero.
        if options is not None:
            if "h_ddot_prescribed" in options:
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
        assert len(self.h_ddot_list) >= int(self.t_max / self.delta_t), "The prescribed vertical acceleration has not enough entries for the whole simulation (starting at t=0)"

        # If there is no prescribed reference lift use the provided function to generate one. If no function was provided, set the reference lift to zero.
        if options is not None:
            if "reference_lift_prescribed" in options:
                self.reference_lift_prescribed = options["reference_lift_prescribed"]
            if "reference_lift_generator" in options:
                self.reference_lift_generator = options["reference_lift_generator"]
        if self.reference_lift_prescribed is not None:
            self.reference_lift_list = np.array(self.reference_lift_prescribed)
        elif self.reference_lift_generator is not None:
            self.reference_lift_list = np.array(self.reference_lift_generator(self))
        else:
            self.reference_lift_list = np.zeros(int(self.t_max / self.delta_t))
            print("No reference lift provided, using zeros instead")
        assert len(self.reference_lift_list) >= int(self.t_max / self.delta_t), "The prescribed reference lift has not enough entries for the whole simulation (starting at t=0)"

        self.h_ddot = self.h_ddot_list[self.time_step]
        self.reference_lift = self.reference_lift_list[self.time_step]
        self.vorticity_field = np.zeros(1)
        self.fy = 0.0
        self.fy_error = self.fy - self.reference_lift
        self.p = np.zeros_like(self.pressure_sensor_positions)
        self.alpha_ddot = 0.0

        # Set up the integrator in Julia
        julia_integrator_reset_commands_template = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_integrator_reset_commands.txt").read_text()
        julia_integrator_reset_commands = julia_integrator_reset_commands_template.format(t_max = self.t_max)
        self.jl.eval(julia_integrator_reset_commands)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render() 

        return observation, info

    def step(self, action):
        assert self.h_ddot is not None, "Call reset before using step method."

        # Assign action to alpha_ddot
        if self.use_discrete_actions:
            new_alpha_ddot = self.discrete_action_values[action] 
        else:
            # Clip and scale
            new_alpha_ddot = min(max(action[0], -1), 1) * self.alpha_ddot_scale

        # compute alpha_ddot differnce for possible penalization
        d_alpha_ddot = new_alpha_ddot - self.alpha_ddot
        self.alpha_ddot = new_alpha_ddot

        # Step system
        # Note that this integrator generally doesn't take the correct timestep (because stop_at_tdt=false). However, this is the only way to avoid discontinuities in the solution
        julia_integrator_step_commands_template = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_integrator_step_commands.txt").read_text()
        julia_integrator_step_commands = julia_integrator_step_commands_template.format(theta_ddot = -self.alpha_ddot, h_ddot = self.h_ddot, n_steps = self.n_solver_steps_per_env_step)
        self.fy = self.jl.eval(julia_integrator_step_commands)
        self.fy_error = self.fy - self.reference_lift

        # For debugging:
        if abs(self.fy) > 1e2:
            pos = self.jl.eval("exogenous_position_vector(aux_state(integrator.u),m,1)")
            vel = self.jl.eval("exogenous_velocity_vector(aux_state(integrator.u),m,1)")
            print("t = {t}".format(t = self.t))
            print("fy = {fy}".format(fy = self.fy))
            print("h_ddot = {h_ddot}".format(h_ddot = self.h_ddot))
            print("alpha_ddot = {alpha_ddot}".format(alpha_ddot = self.alpha_ddot))
            print("pos = {pos}".format(pos = pos))
            print("vel = {vel}".format(vel = vel))

        # Update the time and time step
        # self.t += self.delta_t
        self.t = self.jl.eval("integrator.t")
        self.time_step += 1

        # Check if timelimit is reached
        if self.t > self.t_max or np.isclose(self.t, self.t_max, rtol=1e-9):
            self.truncated = True
        else:
            self.h_ddot = self.h_ddot_list[self.time_step]
            self.reference_lift = self.reference_lift_list[self.time_step]

        # Check if lift goes out of bounds
        if self.lift_termination and (self.fy < self.lift_lower_limit or self.fy > self.lift_upper_limit):
            # self.terminated = True
            self.truncated = True

        # Check if alpha, h_dot, or alpha_dot go out of bounds
        pos = self.jl.eval("exogenous_position_vector(aux_state(integrator.u),m,1)")
        vel = self.jl.eval("exogenous_velocity_vector(aux_state(integrator.u),m,1)")
        if self.alpha_termination and (pos[0] < -self.alpha_limit or pos[0] > self.alpha_limit):
            # self.terminated = True
            self.truncated = True
        if self.alpha_dot_termination and (vel[0] < -self.alpha_dot_limit or vel[0] > self.alpha_dot_limit):
            # self.terminated = True
            self.truncated = True
        if self.h_dot_termination and (vel[1] < -self.h_dot_limit or vel[1] > self.h_dot_limit):
            # self.terminated = True
            self.truncated = True

        # Create observation for next state
        observation = self._get_obs()
        info = self._get_info()
        
        # Compute the reward
        if self.reward_type == 1:
            reward = -((self.fy_error / self.lift_scale) ** 2) # v1
        elif self.reward_type == 2:
            reward = -((self.fy_error / self.lift_scale) ** 2 + 0.1 * (self.alpha_ddot / self.alpha_ddot_scale) ** 2) # v2
        elif self.reward_type == 3:
            reward = -abs(self.fy_error / self.lift_scale) + 1
        elif self.reward_type == 4:
            reward = -abs(self.fy_error / self.lift_scale) - 2 * abs(d_alpha_ddot / self.alpha_ddot_scale) + 1
        elif self.reward_type == 5:
            reward = -1 * (np.exp((self.fy_error / self.lift_scale) ** 2) - 1) + 1 # v5
        elif self.reward_type == 6:
            reward = -np.sqrt(abs(self.fy_error / self.lift_scale)) + 1
        else:
            raise NotImplementedError("Specified reward type is not implemented.")
        
        if self.render_mode == "human":
            self.render() 

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "grayscale_array":
            return self._render_frame()

    def _render_text(self):
        pos = self.jl.eval("exogenous_position_vector(aux_state(integrator.u),m,1)")
        vel = self.jl.eval("exogenous_velocity_vector(aux_state(integrator.u),m,1)")
        outfile = StringIO()
        outfile.write("{:5d}{:10.5f}".format(self.time_step, self.t))
        outfile.write((" {:10.3e}" * 4).format(
            vel[1] / self.h_dot_scale,
            -pos[0] / self.alpha_scale,
            -vel[0] / self.alpha_dot_scale,
            self.h_ddot / self.h_ddot_scale,
        ))
        with closing(outfile):
            return outfile.getvalue()

    def _render_frame(self):
        vorticity_field = np.flip(np.transpose(self.jl.eval("vorticity(integrator).data")),axis=0)
        # quick and dirty mapping from signed float to unsigned int
        vorticity_field = np.round(np.clip(vorticity_field / (2 * self.vorticity_scale) + 0.5, 0, 1) * 255).astype(np.uint8)
        # add extra dim that indicates the image as a single channel (check if this adds extra memory and if it should be made more efficient)
        vorticity_field = np.expand_dims(vorticity_field, axis=2)
        return vorticity_field

