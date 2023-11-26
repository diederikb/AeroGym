update_exogenous!(integrator,[theta_ddot, h_ddot])
fy_last_n_steps = zeros(n_steps)
t_last_n_steps = zeros(n_steps)
for step in 1:n_steps
    step!(integrator)
    _, _, fy_last_n_steps[step] = force(integrator, 1)
    t_last_n_steps[step] = integrator.t
end
(fy_last_n_steps, t_last_n_steps)
