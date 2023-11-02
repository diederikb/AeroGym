update_exogenous!(integrator,[theta_ddot, h_ddot])
for _ in 1:n_steps
    step!(integrator)
end
_, _, fy = force(integrator, 1)
fy
