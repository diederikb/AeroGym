from aero_gym.envs.flow_env import FlowEnv
from gymnasium import spaces
import numpy as np
from julia import Julia
import importlib.resources
from pathlib import Path
import os

# TODO: need to fix the behavior when alpha_init is specified.

class ViscousFlowEnv(FlowEnv):
    """
    The ViscousFlow environment is the two-dimensional, viscous, aerodynamic model for an airfoil undergoing arbitrary motions. The airfoil undergoes prescribed or random vertical accelerations and the goal is to minimize the lift variations by controlling the AOA through the angular acceleration of the airfoil.
    """
    metadata = {"render_modes": ["ansi", "grayscale_array", "grid"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 initialization_time=5.0,
                 initialization_file=None,
                 sys_reinit_commands_file=None,
                 xlim=[-0.75,2.0],
                 ylim=[-0.5,0.5],
                 Re=200,
                 gridRe=4,
                 observe_vorticity_field=False,
                 normalize_vorticity=True,
                 vorticity_scale=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.sys_reinit_commands_file = sys_reinit_commands_file
        self.vorticity_scale = vorticity_scale
        self.observe_vorticity_field = observe_vorticity_field
        self.normalize_vorticity = normalize_vorticity

        # Create the Julia process and set up the viscous flow simulation
        self.jl = Julia()
        self.jl.eval(f"Re={Re}; grid_Re={gridRe}; xmin={xlim[0]}; xmax={xlim[1]}; ymin={ylim[0]}; ymax={ylim[1]}; U={self.U}; c={self.c}; a={self.a}; alpha_init={self.alpha_init}; init_time={initialization_time}; t_max={self.t_max}")
        julia_sys_setup_commands = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_sys_setup_commands.jl").read_text()
        self.jl.eval(julia_sys_setup_commands)

        # Create a flow solution that will be used to initialize the episodes at every reset call. If a file is provided, use the data in there to initialize the flow field. If not, run the flow solver to create an solution
        if initialization_file is not None:
            # TODO
            raise NotImplementedError
        else:
            julia_sys_initialization_commands = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_sys_initialization_with_solver_commands.jl").read_text()
            self.jl.eval(julia_sys_initialization_commands)

        # Get the timestep from julia
        delta_t_solver = self.jl.eval("sys.timestep_func(u0, sys)")
        # For now, make sure that the env delta_t is a multiple of delta_t_solver
        self.n_solver_steps_per_env_step = int(np.floor(self.delta_t / delta_t_solver))
        self.delta_t = self.n_solver_steps_per_env_step * delta_t_solver
        print("Flow solver timestep: delta_t = " + str(delta_t_solver))
        print("Setting environment timestep to multiple of flow solver timestep: delta_t = " + str(self.delta_t))
        
        if self.observe_vorticity_field:
            nx, ny = self.jl.eval("size(zeros_gridcurl(sys))")
            if self.normalize_vorticity: # treat field as a grayscale image
                vorticity_observation_space = spaces.Box(0, 255, shape=(ny, nx, 1), dtype=np.uint8)
            else:
                vorticity_observation_space = spaces.Box(np.inf, -np.inf, shape=(ny, nx, 1), dtype=np.float32)
            self.observation_space = spaces.Dict({"vorticity": vorticity_observation_space, "scalars": self.scalar_observation_space})
        else:
            self.observation_space = self.scalar_observation_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _update_kin_state_attributes(self):
        pos = self.jl.eval("exogenous_position_vector(aux_state(integrator.u),m,1)")
        vel = self.jl.eval("exogenous_velocity_vector(aux_state(integrator.u),m,1)")
        self.alpha = -pos[0]
        self.alpha_dot = -vel[0]
        self.h_dot = vel[1]
        return

    def _get_obs(self):
        if self.observe_previous_pressure:
            p_body = self.jl.eval("pressurejump(integrator).data")
            x_body = self.jl.eval("collect(body)[1]")
            self.p = np.interp(self.pressure_sensor_positions, x_body, p_body)

        scalar_obs = super()._get_obs() 

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        os.system("vmstat")

        # Assign the options to the relevant fields
        if options is not None:
            if "sys_reinit_commands" in options:
                self.sys_reinit_commands_file = options["sys_reinit_commands"]

        self.vorticity_field = np.zeros(1)

        # If there is a file with system reinitialization commands supplied, run them (e.g. to apply forcing)
        if self.sys_reinit_commands_file is not None:
            julia_sys_reinit_commands = Path(self.sys_reinit_commands_file).read_text()
            self.jl.eval(julia_sys_reinit_commands)
            # Recompute n_solver_steps_per_env_step in case the time step changed in the new system
            delta_t_solver = self.jl.eval("sys.timestep_func(u0, sys)")
            self.n_solver_steps_per_env_step = int(np.floor(self.delta_t / delta_t_solver))

        # Set up the integrator in Julia
        julia_integrator_reset_commands = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_integrator_reset_commands.jl").read_text()
        self.jl.eval(julia_integrator_reset_commands)

        _, _, self.fy = self.jl.eval("force(integrator, 1)")
        self.fy_error = self.fy - self.reference_lift

        # Pressure is computed in _get_obs()
        self.p = np.zeros_like(self.pressure_sensor_positions)

        observation = self._get_obs()
        info = super()._get_info()

        if self.render_mode == "human":
            self.render() 

        return observation, info

    def step(self, action):
        assert self.h_ddot is not None, "Call reset before using step method."

        # Assign action to alpha_ddot and compute d_alpha_ddot
        super()._assign_action(action)
        # Update prescribed (or randomly-generated) values
        super()._update_prescribed_values()

        # Step system. Note that we have to take care to take the correct timestep to avoid discontinuities in the solution (this is why we don't use stop_at_tdt)
        self.jl.eval(f"theta_ddot = -({self.alpha_ddot}); h_ddot = {self.h_ddot}; n_steps = {self.n_solver_steps_per_env_step}")
        julia_integrator_step_commands = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_integrator_step_commands.jl").read_text()
        self.fy = self.jl.eval(julia_integrator_step_commands)
        self.fy_error = self.fy - self.reference_lift
        self._update_kin_state_attributes()

		# Update the time and time step
        self.t = self.jl.eval("integrator.t")
        self.time_step += 1

        # For debugging:
        if abs(self.fy) > 1e2:
            print("t = {t}".format(t = self.t))
            print("fy = {fy}".format(fy = self.fy))
            print("h_ddot = {h_ddot}".format(h_ddot = self.h_ddot))
            print("h_dot = {h_dot}".format(h_dot = self.h_dot))
            print("alpha_ddot = {alpha_ddot}".format(alpha_ddot = self.alpha_ddot))
            print("alpha_dot = {alpha_dot}".format(alpha_dot = self.alpha_dot))
            print("alpha = {alpha}".format(alpha = self.alpha))

        # Check if termination or truncation condiations are met
        terminated, truncated = super()._check_termination_truncation()
        
        # Compute the reward
        reward = super()._compute_reward()

        # Create observation for next state
        observation = self._get_obs()
        info = super()._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return super()._render_text()
        if self.render_mode == "grayscale_array":
            return self._render_frame_grayscale_array()
        if self.render_mode == "grid":
            return self._render_frame_grid()

    def _render_frame_grayscale_array(self):
        vorticity_field = np.flip(np.transpose(self.jl.eval("vorticity(integrator).data")),axis=0)
        # quick and dirty mapping from signed float to unsigned int
        vorticity_field = np.round(np.clip(vorticity_field / (2 * self.vorticity_scale) + 0.5, 0, 1) * 255).astype(np.uint8)
        # add extra dim that indicates the image as a single channel (check if this adds extra memory and if it should be made more efficient)
        vorticity_field = np.expand_dims(vorticity_field, axis=2)
        return vorticity_field

    def _render_frame_grid(self):
        vorticity_field = np.transpose(self.jl.eval("vorticity(integrator).data"))
        return vorticity_field
