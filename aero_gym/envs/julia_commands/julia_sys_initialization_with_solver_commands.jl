# Run for a prescribed amount of time to create fully-developed boundary layers
u0 = init_sol(sys)
integrator = init(u0,(0,init_time+1),sys)
step!(integrator, init_time)
u0 = deepcopy(integrator.u)
