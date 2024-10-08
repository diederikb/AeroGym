x_init = init_motion_state(body, m)
update_body!(body, x_init, m)

# quick and dirty way to avoid variable timestep and discontinuities
dt = sys.timestep_func(u0,sys)
t_span = (0.0, t_max + 10 * dt)
# use fully-developed u0
integrator = init(u0,t_span,sys)
