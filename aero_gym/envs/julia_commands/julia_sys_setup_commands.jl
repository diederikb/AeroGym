using ViscousFlow

# setup grid
my_params = Dict()
my_params["Re"] = Re
my_params["grid Re"] = grid_Re
my_params["CFL"] = CFL
my_params["freestream speed"] = U
xlim = (xmin,xmax)
ylim = (ymin,ymax)
g = setup_grid(xlim, ylim, my_params; optimize=false)

# setup airfoil
ds = surface_point_spacing(g, my_params)
body = Plate(1.0, ds)
X = MotionTransform([0.0, 0.0], -(alpha_init))
transform_body!(body, X)

parent_body, child_body = 0, 1
Xp = MotionTransform([0.0, 0.0], 0.0)
Xc = MotionTransform([a, 0.0], 0.0)

adof = ExogenousDOF()
xdof = ConstantVelocityDOF(0)
ydof = ExogenousDOF()
dofs = [adof, xdof, ydof]
joint = Joint(FreeJoint2d, parent_body, Xp, child_body, Xc, dofs)

m = RigidBodyMotion(joint, body)

# Boundary conditions
function my_vsplus(t,x,base_cache,phys_params,motions)
  vsplus = zeros_surface(base_cache)
  surface_velocity_in_translating_frame!(vsplus,x,base_cache,motions,t)
  return vsplus
end

function my_vsminus(t,x,base_cache,phys_params,motions)
  vsminus = zeros_surface(base_cache)
  surface_velocity_in_translating_frame!(vsminus,x,base_cache,motions,t)
  return vsminus
end

bcdict = Dict("exterior" => my_vsplus, "interior" => my_vsminus)

sys = viscousflow_system(g,body,phys_params=my_params,bc=bcdict,motions=m,reference_body=1);
