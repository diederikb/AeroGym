from aero_gym.envs.flow_env import FlowEnv
from gymnasium import spaces
import numpy as np
from juliacall import Main
import importlib.resources
from pathlib import Path
import logging

# TODO: need to fix the behavior when alpha_init is specified.

class ViscousFlowEnv(FlowEnv):
    """
    The ViscousFlow environment is the two-dimensional, viscous, aerodynamic model for an airfoil undergoing arbitrary motions. The airfoil undergoes prescribed or random vertical accelerations and the goal is to minimize the lift variations by controlling the AOA through the angular acceleration of the airfoil.

    ## Arguments
    CFL: CFL number to compute timestep, only considering the constant freestream velocity U. The user needs to adjust for additional forcing in the system, which could increase the velocities beyond their stable limits.
    """
    metadata = {"render_modes": ["ansi", "grayscale_array", "grid"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 initialization_time=5.0,
                 initialization_file=None,
                 sys_reinit_commands=None,
                 xlim=[-0.75,2.0],
                 ylim=[-0.5,0.5],
                 Re=200,
                 grid_Re=4,
                 CFL=0.5,
                 observe_vorticity_field=False,
                 normalize_vorticity=True,
                 vorticity_scale=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.sys_reinit_commands_file = sys_reinit_commands
        self.vorticity_scale = vorticity_scale
        self.observe_vorticity_field = observe_vorticity_field
        self.normalize_vorticity = normalize_vorticity

        # Create the Julia process and set up the viscous flow simulation
        Main.seval(f"Re={Re}; grid_Re={grid_Re}; CFL={CFL}; xmin={xlim[0]}; xmax={xlim[1]}; ymin={ylim[0]}; ymax={ylim[1]}; U={self.U}; c={self.c}; a={self.a}; alpha_init={self.alpha_init}; init_time={initialization_time}; t_max={self.t_max}")
        julia_sys_setup_commands = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_sys_setup_commands.jl").read_text()
        Main.seval(julia_sys_setup_commands)
        self.x_body = Main.seval("collect(body)[1]")

        # Create a flow solution that will be used to initialize the episodes at every reset call. If a file is provided, use the data in there to initialize the flow field. If not, run the flow solver to create an solution
        if initialization_file is not None:
            # TODO
            raise NotImplementedError
        else:
            julia_sys_initialization_commands = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_sys_initialization_with_solver_commands.jl").read_text()
            Main.seval(julia_sys_initialization_commands)

        # Get the timestep from julia
        delta_t_solver = Main.seval("sys.timestep_func(u0, sys)")
        # For now, make sure that the env delta_t is a multiple of delta_t_solver
        self.n_solver_steps_per_env_step = int(np.floor(self.delta_t / delta_t_solver))
        self.delta_t = self.n_solver_steps_per_env_step * delta_t_solver
        logging.info("Flow solver timestep: delta_t = " + str(delta_t_solver))
        logging.info("Setting environment timestep to multiple of flow solver timestep: delta_t = " + str(self.delta_t))
        
        if self.observe_vorticity_field:
            nx, ny = Main.seval("size(zeros_gridcurl(sys))")
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
        pos = Main.seval("exogenous_position_vector(aux_state(integrator.u),m,1)")
        vel = Main.seval("exogenous_velocity_vector(aux_state(integrator.u),m,1)")
        self.alpha = -pos[0]
        self.alpha_dot = -vel[0]
        self.h_dot = vel[1]
        return

    def _get_obs(self):
        scalar_obs = super()._get_obs() 

        if self.observe_vorticity_field:
            vorticity_field = np.flip(np.transpose(Main.seval("vorticity(integrator).data")),axis=0)
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
        
        # Manually call garbage collector to avoid running out of memory due to undiscovered memory leak
        Main.seval("GC.gc()")

        # Assign the options to the relevant fields
        if options is not None:
            if "sys_reinit_commands" in options:
                self.sys_reinit_commands_file = options["sys_reinit_commands"]

        self.vorticity_field = np.zeros(1)

        # If there is a file with system reinitialization commands supplied, run them (e.g. to apply forcing)
        if self.sys_reinit_commands_file is not None:
            julia_sys_reinit_commands = Path(self.sys_reinit_commands_file).read_text()
            Main.seval(julia_sys_reinit_commands)
            # Recompute n_solver_steps_per_env_step in case the time step changed in the new system
            delta_t_solver = Main.seval("sys.timestep_func(u0, sys)")
            self.n_solver_steps_per_env_step = int(np.floor(self.delta_t / delta_t_solver))

        # Set up the integrator in Julia
        julia_integrator_reset_commands = importlib.resources.files("aero_gym").joinpath("envs/julia_commands/julia_integrator_reset_commands.jl").read_text()
        Main.seval(julia_integrator_reset_commands)

        _, _, self.fy = Main.seval("force(integrator, 1)")
        self.fy_error = self.reference_lift - self.fy

        # Set pressure to previous value
        self.p = np.zeros_like(self.pressure_sensor_positions)

        # Not ideal that this sets the pressure to the latest available value, because this is not the behavior in step().
        # However, this is not an issue for alpha_init = 0, because the pressure jump will be zero
        if self.observe_previous_pressure:
            p_body = Main.seval("pressurejump(integrator).data")
            self.p = np.interp(self.pressure_sensor_positions, self.x_body, p_body)

        observation = self._get_obs()
        info = super()._get_info()

        if self.render_mode == "human":
            self.render() 

        return observation, info

    def step(self, action):
        assert self.h_ddot is not None, "Call reset before using step method."

        # Assign action to alpha_ddot and compute d_alpha_ddot
        super()._assign_action(action)

        # For debugging:
        if abs(self.fy) > 1e2:
            print("t = {t}".format(t = self.t))
            print("fy = {fy}".format(fy = self.fy))
            print("h_ddot = {h_ddot}".format(h_ddot = self.h_ddot))
            print("h_dot = {h_dot}".format(h_dot = self.h_dot))
            print("alpha_ddot = {alpha_ddot}".format(alpha_ddot = self.alpha_ddot))
            print("alpha_dot = {alpha_dot}".format(alpha_dot = self.alpha_dot))
            print("alpha = {alpha}".format(alpha = self.alpha))

        # Transfer alpha_ddot and h_ddot to Julia and step integrator for n_solver_steps_per_env_step
        Main.seval(f"update_exogenous!(integrator,[-({self.alpha_ddot}), {self.h_ddot}])")
        for step in range(self.n_solver_steps_per_env_step):
            Main.seval("step!(integrator)")
            if step == self.n_solver_steps_per_env_step - 1:
                # Set lift and pressure to the latest computed value with the latest alpha_ddot and h_ddot.
                # Ideally, we set the lift and pressure to first computed value with the latest alpha_ddot and h_ddot. This matches the behavior of the wagner env the closest.
                (mom, _, fy) = Main.seval("force(integrator, 1)")
                self.fy = fy
                if self.observe_previous_pressure:
                    p_body = Main.seval("pressurejump(integrator).data")
                    self.p = np.interp(self.pressure_sensor_positions, self.x_body, p_body)

        # Compute lift errors with reference
        self.fy_error = self.reference_lift - self.fy
        self.fy_integrated_error = self.fy_integrated_error + self.fy_error * self.delta_t

        # Update histories
        super()._update_hist()

	# Update the time and time step
        self.t = Main.seval("integrator.t")
        self.time_step += 1

        # Update kin states to latest available values
        self._update_kin_state_attributes()

        # Check if termination or truncation condiations are met
        terminated, truncated = super()._check_termination_truncation()
        
        # Compute the reward
        reward = super()._compute_reward()

        # Update prescribed (or randomly-generated) values
        super()._update_prescribed_values()

        # Create observation for next state
        observation = self._get_obs()
        info = super()._get_info()

        if terminated or truncated:
            (solver_mom_hist, _, solver_fy_hist) = Main.seval("force(integrator.sol, sys, 1)")
            solver_mom_hist = np.array(solver_mom_hist)
            solver_fy_hist = np.array(solver_fy_hist)
            solver_alpha_hist = np.array(Main.seval("map(u -> -exogenous_position_vector(aux_state(u), m, 1)[1], integrator.sol)"))
            solver_alpha_dot_hist = np.array(Main.seval("map(u -> -exogenous_velocity_vector(aux_state(u), m, 1)[1], integrator.sol)"))
            solver_h_dot_hist = np.array(Main.seval("map(u -> exogenous_velocity_vector(aux_state(u), m, 1)[2], integrator.sol)"))
            solver_t_hist = np.array(Main.seval("integrator.sol.t"))
            solver_vort_hist = np.array(Main.seval("map(u -> u.data, vorticity(integrator.sol, sys, integrator.sol.t))"))
            info["solver_fy_hist"] = solver_fy_hist
            info["solver_power_hist"] = solver_mom_hist * solver_alpha_dot_hist
            info["solver_alpha_hist"] = solver_alpha_hist
            info["solver_alpha_dot_hist"] = solver_alpha_dot_hist
            info["solver_h_dot_hist"] = solver_h_dot_hist
            info["solver_t_hist"] = solver_t_hist - self.delta_t
            info["solver_vort_hist"] = solver_vort_hist

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return super()._render_text()
        if self.render_mode == "grayscale_array":
            return self._render_frame_grayscale_array()
        if self.render_mode == "grid":
            return self._render_frame_grid()

    def _render_frame_grayscale_array(self):
        vorticity_field = np.flip(np.transpose(Main.seval("vorticity(integrator).data")),axis=0)
        # quick and dirty mapping from signed float to unsigned int
        vorticity_field = np.round(np.clip(vorticity_field / (2 * self.vorticity_scale) + 0.5, 0, 1) * 255).astype(np.uint8)
        # add extra dim that indicates the image as a single channel (check if this adds extra memory and if it should be made more efficient)
        vorticity_field = np.expand_dims(vorticity_field, axis=2)
        return vorticity_field

    def _render_frame_grid(self):
        vorticity_field = np.transpose(Main.seval("vorticity(integrator).data"))
        return vorticity_field
