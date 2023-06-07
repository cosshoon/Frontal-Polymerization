# Michael Zakoworotny, Gavin DeBrun
# Thermo-chemical model for printing of continuous fiber composites,
# where impregnated composite tow is extruded through rollers
# Steady model
# Using nondimensionalization based on: 
# Char. length = diffusion / front velocity (~ front width)
# Char. velocity = front velocity (for a first order reaction)
# Char. time = diffusion / front velocity ** 2


from dolfin import *
import os
import sys
import time
from math import ceil, floor, pi, tanh, sqrt, acos, cos
import numpy as np
# import gmsh
import meshio
from mpi4py import MPI
import csv
from ufl import tanh as tanh_ufl
from scipy.interpolate import interp1d
# from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

# Print message on only one process
def print_parallel(mssg):
    if comm_rank==0:
        print(mssg)
        sys.stdout.flush()

set_log_level(40) # error level=40, warning level=30, info level=20
# Switch to directory of file
try:
    os.chdir(os.path.dirname(__file__))
except:
    print_parallel("Cannot switch directories")

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

QUAD_DEG = 4
parameters["form_compiler"]["quadrature_degree"] = QUAD_DEG
dx = dx(degree=QUAD_DEG, scheme="default")

######################################### PARAMETERS ##########################################

# For 3K,6K,12K tow:
# A_tow_3K =  0.12e-6 # cross section area in m^2 of Hexcel AS4 tow
# A_tow_6K =  0.24e-6
# A_tow_12K = 0.48e-6
# aspect_tow = 1 # guess of aspect ratio of tow (width to thickness)
# H_tow = sqrt(A_tow_12K / aspect_tow) # initial thickness of impregnated tow in m (before passing through roller)
# B_tow = H_tow*aspect_tow # initial width of impregnated tow in m
# H_tow = 1e-3 
phi_1 = 0.3 # initial fiber volume fraction of impregnated tow (before compaction)
phi_2_target = 0.4

# Geometry
R_r = 40e-3 # estimate roller diameter as 80 mm
L_up = 1e-3 # length of tow upstream of left roller contact point (ie. domain starts at x = -(L_up + l_r) )
L_down = 10e-3 # length of tow downstream of roller contact point
L = L_up + L_down # total length of tow domain

# Process parameters (controlled by printing system)
om_r = 0.075 # rotational speed of roller
V_r = om_r * R_r # linear speed of surface of roller / tow
T_r = 180 + 273.15 # temperature of roller (assuming perfect contact between roller and tow)
H_gap = 2*0.52e-3 # gap spacing between rollers

H_tow = phi_2_target / phi_1 * H_gap


# Derived quantities
phi_2 = H_tow/H_gap * phi_1 # compacted fiber volume fraction
dH = (H_tow - H_gap)/2 # amount of compaction of tow (on each side of roller)
a_r = acos(1 - dH/R_r) # contact angle (in radians) between the roller and the tow
# a_r = acos(1 - H_gap/(2*R_r)*(phi_2/phi_1 - 1))
l_r = a_r * R_r # linear contact length between roller and tow

# DCPD resin properties
# Thermo-chemical parameters
rhor = 980
kr =  0.15
Cpr = 1600.0
R_ = 8.314
# Nature values
Hr =  350000
A_ = 8.55e15
Er = 110750.0
n_ = 1.72
m_ = 0.77
Ca = 14.48
alpha_c = 0.41

# HexTow AS4 Carbon Fiber
rhof = 1790.0
kf = 6.83
Cpf = 1129.0

# Initial conditions of tow
T0 = 20 + 273.15 # initial temperature of tow
alpha0 = 0.1 # initial degree of cure of resin in impregnated tow

# Environmental conditions (convection)
h_conv = 50 # heat convection coefficient
Tamb = 30 + 273.15 # ambient temperature

# Steady solution parameters
# loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # load ramping values (ramping roller temperature)
load_dT = 10 # gap spacing in temperature between loads
loads = (np.arange(T0, T_r+load_dT, load_dT) - T0 ) / (T_r - T0)
iter_tol = 1e-10 # tolerance on steady solution
max_iter = 60 # maximum number of iterations per load step

# Homogenized properties - as functions of input phi
k_par_fun = lambda p : kr*(1-p) + kf*p # thermal cond. parallel to fibers (x axis)
k_perp_fun = lambda p : kr + 2*p*kr / ((kf+kr)/(kf-kr) - p + (kf-kr)/(kf+kr)*(0.30584*p**4 + 0.013363*p**8)) # thermal cond. perpendicular to fibers (y axis)
rho_bar_fun = lambda p : rhor*(1-p) + rhof*p # homogenized density
Cp_bar_fun = lambda p : (Cpr*rhor*(1-p) + Cpf*rhof*p)/rho_bar_fun(p) # homogenized specific heat
rho_Cp_bar_fun = lambda p : Cpr*rhor*(1-p) + Cpf*rhof*p # product of homogenized density and specific heat

# Characteristic scales
phi0 = phi_2 # use phi2 as the reference fiber volume fraction
T_max = T0 + rhor*Hr/rho_Cp_bar_fun(phi0)*(1-phi0)*(1-alpha0)
v_c = (A_*R_*k_par_fun(phi0)*T_max**2/rhor/Hr/Er/(1-phi0)/(1-alpha0) * np.exp(-Er/R_/T_max))**(1/2)
L_c = k_par_fun(phi0)/rho_Cp_bar_fun(phi0)/v_c
t_c = L_c/v_c
T_c = T_max

# Nondimensional properties
L_up_star = L_up/L_c
L_down_star = L_down/L_c
l_r_star = l_r/L_c
H_gap_star = H_gap/L_c
H_tow_star = H_tow/L_c
R_r_star = R_r/L_c
theta_r = (T_r - T0)/(T_c - T0)
theta_amb = (Tamb - T0)/(T_c - T0)
v_star = V_r / v_c

# Nondimensional groups
k_bar_0 = as_tensor([[k_par_fun(phi0), 0],[0, k_perp_fun(phi0)]])
Fo_bar = as_tensor([[1, 0],[0, k_perp_fun(phi0)/k_par_fun(phi0)]])   #1/L_c/v_c/rho_Cp_bar_fun(phi0) * k_bar_0 # tensor form
H_bar = 1/(1-alpha0)
Z = rhor*Hr*Er*(1-phi0)*(1-alpha0)/rho_Cp_bar_fun(phi0)/R_/T_max**2
delta = Er/R_/T_max
gamma = T0/(T_max - T0)

################################### SETUP PROBLEM ######################################

h_elem = 2e-5/L_c
# nel_x = ceil(L/h_elem)
# nel_y = ceil((H_gap/2)/h_elem)

# Generate mesh (use gmsh to ensure points at roller contact points)
# def generate_roller_mesh(h):
#     # Number of elements
#     nel_y = ceil((H_gap_star/2)/h)
#     nel_x_up = floor(L_up_star/h)
#     nel_x_roll = ceil(l_r_star/h)
#     nel_x_down = ceil(L_down_star/h)

#     # Mesh geometry
#     mesh_size = 0.1
#     mesh_name = "roller_mesh"
#     gmsh.initialize()
#     gmsh.model.add(mesh_name)
#     # Points
#     p1 = gmsh.model.geo.addPoint(-(L_up_star+l_r_star),0,0,mesh_size) # bottom points
#     p2 = gmsh.model.geo.addPoint(-l_r_star,0,0,mesh_size)
#     p3 = gmsh.model.geo.addPoint(0,0,0,mesh_size)
#     p4 = gmsh.model.geo.addPoint(L_down_star,0,0,mesh_size)
#     p5 = gmsh.model.geo.addPoint(-(L_up_star+l_r_star),H_gap_star/2,0,mesh_size) # bottom points
#     p6 = gmsh.model.geo.addPoint(-l_r_star,H_gap_star/2,0,mesh_size)
#     p7 = gmsh.model.geo.addPoint(0,H_gap_star/2,0,mesh_size)
#     p8 = gmsh.model.geo.addPoint(L_down_star,H_gap_star/2,0,mesh_size)
#     # Lines
#     c1 = gmsh.model.geo.addLine(p1,p2) # bottom horiz lines
#     c2 = gmsh.model.geo.addLine(p2,p3)
#     c3 = gmsh.model.geo.addLine(p3,p4)
#     c4 = gmsh.model.geo.addLine(p1,p5) # vert Lines
#     c5 = gmsh.model.geo.addLine(p4,p8)
#     c6 = gmsh.model.geo.addLine(p5,p6) # upper horiz lines
#     c7 = gmsh.model.geo.addLine(p6,p7)
#     c8 = gmsh.model.geo.addLine(p7,p8)
#     # Surfaces
#     cl1 = gmsh.model.geo.addCurveLoop([c1,c2,c3,c5,-c8,-c7,-c6,-c4])
#     s1 = gmsh.model.geo.addPlaneSurface([cl1])
#     # Meshing setup
#     gmsh.model.geo.mesh.setTransfiniteSurface(s1, cornerTags=[p1,p4,p8,p5])
#     gmsh.model.geo.mesh.setTransfiniteCurve(c1, nel_x_up+1, coef=1)
#     gmsh.model.geo.mesh.setTransfiniteCurve(c2, nel_x_roll+1, coef=1)
#     gmsh.model.geo.mesh.setTransfiniteCurve(c3, nel_x_down+1, coef=1)
#     gmsh.model.geo.mesh.setTransfiniteCurve(c5, nel_y+1, coef=1)
#     gmsh.model.geo.mesh.setTransfiniteCurve(c8, nel_x_down+1, coef=1)
#     gmsh.model.geo.mesh.setTransfiniteCurve(c7, nel_x_roll+1, coef=1)
#     gmsh.model.geo.mesh.setTransfiniteCurve(c6, nel_x_up+1, coef=1)
#     gmsh.model.geo.mesh.setTransfiniteCurve(c4, nel_y+1, coef=1)

#     # Mesh and write to file
#     gmsh.model.geo.synchronize()
#     gmsh.model.mesh.generate(2)
#     gmsh.write(mesh_name+".msh")
#     # Import gmsh mesh - first write msh to xdmf, then import to fenics
#     if comm_rank==0:
#         msh = meshio.read(mesh_name+".msh")
#         msh.prune_z_0()
#         meshio.write(mesh_name+".xdmf", meshio.Mesh(points=msh.points, cells={"triangle":msh.get_cells_type("triangle")})) # add cell data if physical groups specified
#     mesh = Mesh()
#     with XDMFFile(mesh_name+".xdmf") as infile:
#         infile.read(mesh)
#     if comm_rank==0:
#         os.remove(mesh_name+".msh")
#         os.remove(mesh_name+".xdmf")
#         os.remove(mesh_name+".h5")

#     return mesh


# mesh = generate_roller_mesh(h_elem)
mesh = RectangleMesh(Point(-(L_up_star+l_r_star), 0), Point(L_down_star, H_gap_star/2), ceil((L_up_star+l_r_star+L_down_star)/h_elem), ceil((H_gap_star/2)/h_elem), "right")
dx = Measure('dx', domain=mesh)

# Define boundaries
left = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=-(L_up_star+l_r_star), tol=1e-7)
bot = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=0.0, tol=1e-7)
right = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=L_down_star, tol=1e-7)
top_down = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=x2-tol", side=H_gap_star/2, x2=0, tol=1e-7)
top_roll = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=x1-tol && x[0]<=x2+tol", side=H_gap_star/2, x1=-l_r_star, x2=0, tol=1e-7)
top_up = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]<=x1+tol", side=H_gap_star/2, x1=-l_r_star, tol=1e-7)
# Mark boundaries
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
facets.set_all(0)
left.mark(facets, 1)
bot.mark(facets, 2)
right.mark(facets, 3)
top_down.mark(facets, 4)
top_roll.mark(facets, 5)
top_up.mark(facets, 6)
ds = Measure('ds', domain=mesh, subdomain_data=facets) # partition boundary
# ds = ds(subdomain_data=facets)

# Function spaces for thermo-chemical problem
V_t = FunctionSpace(mesh, "CG", 1)
w = TestFunction(V_t)
theta = TrialFunction(V_t)
theta_ = Function(V_t, name="theta")
T_ = Function(V_t, name="T_[C]")
V_a = FunctionSpace(mesh, "CG", 1)
beta = TestFunction(V_a)
dal = TrialFunction(V_a)
al = Function(V_a, name="alpha")
al_last = Function(V_a) # alpha at previous iteration
al_diff = Function(V_a) # difference in alpha from previous iteration

# Boundary conditions
T_roller = Expression("theta", degree=1, theta=theta_r)
bc_roller = DirichletBC(V_t, T_roller, facets, 5) # temperature at roller
T_in = Expression("theta", degree=1, theta=0)
bc_T_in = DirichletBC(V_t, T_in, facets, 1) # initial temperature of tow
bcs_T = [bc_roller, bc_T_in]

alpha_in = Expression("a", degree=1, a=alpha0)
bc_a_in = DirichletBC(V_a, alpha_in, facets, 1) # initial degree of cure of tow
bcs_a = [bc_a_in]

# Initial conditions
T_init = Expression("theta", degree=1, theta=0)
theta_.assign(project(T_init, V_t))
alpha_init = Expression("a", degree=1, a=alpha0)
al.assign(project(alpha_init, V_a))
al_last.assign(project(alpha_init, V_a))

# Parameters for numerical model
stablz = CellDiameter(mesh)
x = MeshCoordinates(mesh)
n = FacetNormal(mesh)

v_ = Expression(("v","0."), degree=1, v=v_star) # Velocity field

# phi = Expression() # fiber volume fraction field
class FiberVolFrac(UserExpression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def eval(self,values,x):
        if x[0] <= -l_r_star:
            phi = phi_1
        elif x[0] <= 0:
            phi = H_tow_star*phi_1 / (H_tow_star - 2*(sqrt(R_r_star**2 - x[0]**2) - R_r_star*cos(a_r)))
        else:
            phi = phi_2
        values[0] = phi

    def value_shape(self):
        return ()

phi = FiberVolFrac()

# Homogenized properties
k_par = k_par_fun(phi) # thermal cond. parallel to fibers (x axis)
k_perp = k_perp_fun(phi) # thermal cond. perpendiculat to fibers (y axis)
# k_tens = k_tens_fun(phi) # thermal conductivity tensor
rho_bar = rho_bar_fun(phi) # homogenized density
Cp_bar = Cp_bar_fun(phi) # homogenized specific heat
rho_Cp_bar = rho_Cp_bar_fun(phi) # product of homogenized density and specific heat
# Ratios for fiber volume fraction dependence of homogenized properties
rho_Cp_til = rho_Cp_bar/rho_Cp_bar_fun(phi0)
k_til = as_tensor([[k_par/k_par_fun(phi0), 0],[0, k_perp/k_perp_fun(phi0)]])
one_m_phi_til = (1 - phi)/(1 - phi0)
rho_Cp_til_fun = Function(V_t) # need to project rho_Cp_til to a function to be able to take gradient of it
rho_Cp_til_fun.assign(project(rho_Cp_til, V_t))

# Variational form for degree of cure problem - NONDIMENSIONAL
tau_a = stablz/v_star
g = (1-al)**n_ * al**m_ / (1+exp(Ca*(al-alpha_c)))
F_a = (beta + tau_a*dot(v_, grad(beta)))*(dot(v_, grad(al)) - Z*exp(delta*(theta_ - 1)/(theta_ + gamma))*g)*dx
tang_a = derivative(F_a, al, dal)

# Variational form for thermal problem
Pe = v_star*v_c*L_c*stablz/2/(k_par/rho_Cp_bar) # Peclet number for thermal problem (diffusivity in x direction)
tau_T = stablz/v_star * (1/tanh_ufl(Pe) - 1/Pe)
# ONLY APPLYING SUPG TEST FN TO MATERIAL DERIVATIVE TERMS
# F_T = dot(grad(w),dot(k_tens, grad(T)))*dx + (w + tau_T*dot(v_, grad(w)))*(rho_Cp_bar*dot(v_, grad(T)) \
#                                                         - rhor*Hr*(1-phi)*dot(v_, grad(al)))*dx + w*h_conv*(T - Tamb)*ds(4)
F_T = 1/rho_Cp_til*dot(dot(Fo_bar , grad(w)) , dot(k_til , grad(theta)))*dx  +  w*dot(dot(Fo_bar , grad(1/rho_Cp_til_fun)) , dot(k_til, grad(theta)))*dx \
        + (w + tau_T*dot(v_, grad(w)))*(dot(v_, grad(theta)) - H_bar*one_m_phi_til/rho_Cp_til*dot(v_, grad(al)))*dx \
        + h_conv/v_c/rho_Cp_bar_fun(phi0)/rho_Cp_til*w*(theta - theta_amb)*ds(4)

# Solver for degree of cure
# problem_alpha = LinearVariationalProblem(lhs(F_a), rhs(F_a), al_, bcs_a, form_compiler_parameters=ffc_options)
# solver_alpha = LinearVariationalSolver(problem_alpha)
problem_alpha = NonlinearVariationalProblem(F_a, al, bcs=bcs_a, J=tang_a, form_compiler_parameters=ffc_options)
solver_alpha = NonlinearVariationalSolver(problem_alpha)
prms_a = solver_alpha.parameters
prms_a["nonlinear_solver"] = "newton"
prms_a["newton_solver"]["report"] = True
prms_a["newton_solver"]["absolute_tolerance"] = 1e-14
prms_a["newton_solver"]["relative_tolerance"] = 1e-14
prms_a["newton_solver"]["error_on_nonconvergence"] = False
prms_a["newton_solver"]["convergence_criterion"] = "residual"
prms_a["newton_solver"]["maximum_iterations"] = 8
prms_a["newton_solver"]["relaxation_parameter"] = 1.
prms_a["newton_solver"]["linear_solver"] = "gmres"
prms_a["newton_solver"]["preconditioner"] = "hypre_amg"
prms_a["newton_solver"]['krylov_solver']['absolute_tolerance'] = 1e-12
prms_a["newton_solver"]['krylov_solver']['relative_tolerance'] = 1e-12
prms_a["newton_solver"]['krylov_solver']['maximum_iterations'] = 100
prms_a["newton_solver"]['krylov_solver']['monitor_convergence'] = False

# Solver for temperature
problem_temp = LinearVariationalProblem(lhs(F_T), rhs(F_T), theta_, bcs_T, form_compiler_parameters=ffc_options)
solver_temp = LinearVariationalSolver(problem_temp)
prms_T = solver_temp.parameters
prms_T["linear_solver"] = "gmres"
prms_T["preconditioner"] = "hypre_amg"
prms_T['krylov_solver']['absolute_tolerance'] = 1e-12
prms_T['krylov_solver']['relative_tolerance'] = 1e-12
prms_T['krylov_solver']['maximum_iterations'] = 100

################################### SOLUTION #################################################

# Output file
file_results = XDMFFile("results_comp_print_nonDim.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

file_results.write(theta_, 0)
file_results.write(al, 0)

# Write functions for parameters that vary with fiber volume fraction (write the ratios, tilde parameters)
V_tens = TensorFunctionSpace(mesh, "CG", 1)
k_fun = Function(V_tens, name="therm_cond_norm")
k_fun.assign(project(k_til, V_tens))
file_results.write(k_fun, 0)

phi_fun = Function(V_t, name="fiber_vol_frac")
phi_fun.assign(project(phi, V_t))
file_results.write(phi_fun, 0)

rho_Cp_fun = Function(V_t, name="rho_Cp_bar_norm")
rho_Cp_fun.assign(project(rho_Cp_til, V_t))
file_results.write(rho_Cp_fun, 0)

# Get dofs for midline
alpha_coords = V_a.tabulate_dof_coordinates()
alpha_dof = np.array(V_a.dofmap().dofs()) # dofs for alpha among all processors
alpha_midline_dofs_local = np.where(alpha_coords[:,1] <= 0)[0] # midline dofs, using this core's dofs
alpha_midline_dofs = alpha_dof[alpha_midline_dofs_local] # midline dofs, using global dofs

theta_roller_list = []
alpha_max_list = []
front_loc_list = []
Q_roller_list = []

load_prev = 0
# Iterative solution for steady problem
for l_iter, load in enumerate(loads):
    
    print_parallel("Solving for load {}".format(load))

    # Update bounday condition for load
    T_roller.theta = load*theta_r # ramp roller temperature

    for k in range(1, max_iter+1):
        solver_temp.solve()
        (it_nl, conv_nl) = solver_alpha.solve()

        # If solution for alpha does not converge, use alternative solver
        if conv_nl != 1:
            print_parallel("Not converged")
            
            al.assign(al_last)

            x_ = Function(V_a) # solution vector 
            tol_abs_nl = 1e-12
            tol_rel_nl = 1e-4 # checks if change from last iteration is small
            tol_inc_nl = 1e-13
            max_iter_nl = 8
            bcs_a_hold = [DirichletBC(V_a, Constant(0.), facets, 1)]

            it = 0
            # Apply dirichlet BC on first iteration only (not necessary since initial guess always satisfies dirichlet BC)
            # if k == 0:
            #     it += 1
            #     A = Matrix()
            #     b = Vector()
            #     sa = SystemAssembler(tang_a, -F_a, bcs_du_init)
            #     sa.assemble(A, b)
            #     solve(A, x_.vector(), b)
            #     error = b.norm('l2')
            #     error_inc = norm(x_)
            #     print_parallel('Custom newton iteration: {:d}; Residual: {:.3e}; Incremental: {:.3e} '.format(it, error, error_inc))
            #     al.vector()[:] += x_.vector()[:]
                
            #     al.vector()[:] = np.maximum(al.vector()[:], alpha0-1e-4)
            #     al.vector()[:] = np.minimum(al.vector()[:], 0.999)
            # else:
            error = 1
            error_last = 1
            error_rel = 1
            error_inc = 1

            while error > tol_abs_nl and error_rel > tol_rel_nl and error_inc > tol_inc_nl and it <= max_iter_nl:
                it += 1
                A, b = assemble_system(tang_a, -F_a, bcs_a_hold)
                solve(A, x_.vector(), b)
                error = b.norm('l2')
                error_rel = abs(error-error_last)/error_last #if it > 1 else 1
                error_inc = norm(x_)
                error_last = error
                print_parallel('Iteration: {:d}; Residual: {:.3e}; Relative: {:.3e}, Incremental: {:.3e} '.format(it, error, error_rel, error_inc))
                al.vector()[:] += x_.vector()[:]

                # file_results.write(al, load_prev + (load-load_prev)*(k-1 + it/max_iter_nl)/max_iter)

                al.vector()[:] = np.maximum(al.vector()[:], alpha0-1e-4)
                al.vector()[:] = np.minimum(al.vector()[:], 1-1e-4)

        al.vector()[:] = np.minimum(al.vector()[:],0.999)
        al.vector()[:] = np.maximum(al.vector()[:],alpha0-1e-4)

        al_diff.vector()[:] = al.vector()[:] - al_last.vector()[:]
        it_norm = norm(al_diff)
        print_parallel("\t Iteration {}, residual {} (tolerance {})".format(k, it_norm, iter_tol))

        al_last.assign(al)

        # Write solution for current iteration
        # file_results.write(T_, load_prev + (load-load_prev)*k/max_iter)
        # file_results.write(al, load_prev + (load-load_prev)*k/max_iter)

        # Check if solution converge on this iteration
        if it_norm <= iter_tol:
            print_parallel("\t Converged for iteration {}".format(k))
            break

    # Get midline alpha values, sorted from left (x=-(L_up+l_r)) to right (x=L_down)
    x_coord_gather = comm.gather(alpha_coords[alpha_midline_dofs_local,0], root=0)
    alpha_gather = comm.gather(al.vector()[alpha_midline_dofs_local], root=0)
    x_coord_sorted = []
    alpha_sorted = []
    if comm_rank == 0:
        x_coord_gathered = np.concatenate(x_coord_gather)
        alpha_gathered = np.concatenate(alpha_gather)
        x_coord_sort_inds = x_coord_gathered.argsort()
        x_coord_sorted = x_coord_gathered[x_coord_sort_inds]
        alpha_sorted = alpha_gathered[x_coord_sort_inds]
    x_midline = comm.bcast(x_coord_sorted, root=0)
    alpha_midline = comm.bcast(alpha_sorted, root=0)

    alpha_max = max(alpha_midline)
    try:
        front_loc = interp1d(alpha_midline, x_midline)(0.5)
    except:
        front_loc = -np.inf

    theta_roller_list.append(load*theta_r)
    alpha_max_list.append(alpha_max)
    front_loc_list.append(front_loc)

    T_.assign(project((T_max - T0)*theta_+T0 - 273.15, V_t))

    # Compute energy exchange with roller (power per unit thickness, from ONE roller)
    q_roller = (T_c - T0)/L_c * assemble(-dot(dot(dot(k_bar_0,k_til),grad(theta_)),n)*ds(5))
    Q_roller_list.append(q_roller)
    
    # Write converged solution for load step
    file_results.write(theta_, load)
    file_results.write(al, load)
    file_results.write(T_, load)

    
    load_prev = load

print_parallel("")
print_parallel("Roller contact length: {} m".format(l_r))
print_parallel("Load no\ttheta_r\talpha_max\tfront_x_star\tQ_roller (from both rollers, power per unit thickness [W/m])")
for i, (theta_r_i, alpha_max_i, front_loc_i, q_roller_i) in enumerate(zip(theta_roller_list, alpha_max_list, front_loc_list, Q_roller_list)):
    print_parallel("{}\t{}\t{}\t{}\t{}".format(i, theta_r_i, alpha_max_i, front_loc_i, 2*q_roller_i))
