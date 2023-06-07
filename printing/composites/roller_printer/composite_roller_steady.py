# Michael Zakoworotny, Gavin DeBrun
# Thermo-chemical model for printing of continuous fiber composites,
# where impregnated composite tow is extruded through rollers
# Steady model


from dolfin import *
import os
import sys
import time
from math import ceil, floor, pi, tanh, sqrt, acos, cos
import numpy as np
import gmsh
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

# set_log_level(30) # error level=40, warning level=30, info level=20
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
L_down = 5e-3 # length of tow downstream of roller contact point
L = L_up + L_down # total length of tow domain

# Process parameters (controlled by printing system)
om_r = 0.05 # rotational speed of roller
V_r = om_r * R_r # linear speed of surface of roller / tow
T_r = 100 + 273.15 # temperature of roller (assuming perfect contact between roller and tow)
H_gap = 0.52e-3 # gap spacing between rollers

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
loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # load ramping values (ramping roller temperature)
iter_tol = 1e-10 # tolerance on steady solution
max_iter = 150 # maximum number of iterations per load step

################################### SETUP PROBLEM ######################################

h_elem = 2e-5
# nel_x = ceil(L/h_elem)
# nel_y = ceil((H_gap/2)/h_elem)

# Generate mesh (use gmsh to ensure points at roller contact points)
def generate_roller_mesh(h):
    # Number of elements
    nel_y = ceil((H_gap/2)/h)
    nel_x_up = floor(L_up/h)
    nel_x_roll = ceil(l_r/h)
    nel_x_down = ceil(L_down/h)

    # Mesh geometry
    mesh_size = 0.1
    mesh_name = "roller_mesh"
    gmsh.initialize()
    gmsh.model.add(mesh_name)
    # Points
    p1 = gmsh.model.geo.addPoint(-(L_up+l_r),0,0,mesh_size) # bottom points
    p2 = gmsh.model.geo.addPoint(-l_r,0,0,mesh_size)
    p3 = gmsh.model.geo.addPoint(0,0,0,mesh_size)
    p4 = gmsh.model.geo.addPoint(L_down,0,0,mesh_size)
    p5 = gmsh.model.geo.addPoint(-(L_up+l_r),H_gap/2,0,mesh_size) # bottom points
    p6 = gmsh.model.geo.addPoint(-l_r,H_gap/2,0,mesh_size)
    p7 = gmsh.model.geo.addPoint(0,H_gap/2,0,mesh_size)
    p8 = gmsh.model.geo.addPoint(L_down,H_gap/2,0,mesh_size)
    # Lines
    c1 = gmsh.model.geo.addLine(p1,p2) # bottom horiz lines
    c2 = gmsh.model.geo.addLine(p2,p3)
    c3 = gmsh.model.geo.addLine(p3,p4)
    c4 = gmsh.model.geo.addLine(p1,p5) # vert Lines
    c5 = gmsh.model.geo.addLine(p4,p8)
    c6 = gmsh.model.geo.addLine(p5,p6) # upper horiz lines
    c7 = gmsh.model.geo.addLine(p6,p7)
    c8 = gmsh.model.geo.addLine(p7,p8)
    # Surfaces
    cl1 = gmsh.model.geo.addCurveLoop([c1,c2,c3,c5,-c8,-c7,-c6,-c4])
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    # Meshing setup
    gmsh.model.geo.mesh.setTransfiniteSurface(s1, cornerTags=[p1,p4,p8,p5])
    gmsh.model.geo.mesh.setTransfiniteCurve(c1, nel_x_up+1, coef=1)
    gmsh.model.geo.mesh.setTransfiniteCurve(c2, nel_x_roll+1, coef=1)
    gmsh.model.geo.mesh.setTransfiniteCurve(c3, nel_x_down+1, coef=1)
    gmsh.model.geo.mesh.setTransfiniteCurve(c5, nel_y+1, coef=1)
    gmsh.model.geo.mesh.setTransfiniteCurve(c8, nel_x_down+1, coef=1)
    gmsh.model.geo.mesh.setTransfiniteCurve(c7, nel_x_roll+1, coef=1)
    gmsh.model.geo.mesh.setTransfiniteCurve(c6, nel_x_up+1, coef=1)
    gmsh.model.geo.mesh.setTransfiniteCurve(c4, nel_y+1, coef=1)

    # Mesh and write to file
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_name+".msh")
    # Import gmsh mesh - first write msh to xdmf, then import to fenics
    if comm_rank==0:
        msh = meshio.read(mesh_name+".msh")
        msh.prune_z_0()
        meshio.write(mesh_name+".xdmf", meshio.Mesh(points=msh.points, cells={"triangle":msh.get_cells_type("triangle")})) # add cell data if physical groups specified
    mesh = Mesh()
    with XDMFFile(mesh_name+".xdmf") as infile:
        infile.read(mesh)
    if comm_rank==0:
        os.remove(mesh_name+".msh")
        os.remove(mesh_name+".xdmf")
        os.remove(mesh_name+".h5")

    return mesh


# mesh = generate_roller_mesh(h_elem)
mesh = RectangleMesh(Point(-(L_up+l_r), 0), Point(L_down, H_gap/2), ceil((L_up+l_r+L_down)/h_elem), ceil((H_gap/2)/h_elem), "right")
dx = Measure('dx', domain=mesh)

# Define boundaries
left = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=-(L_up+l_r), tol=1e-7)
bot = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=0.0, tol=1e-7)
right = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=L_down, tol=1e-7)
top_down = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=x2-tol", side=H_gap/2, x2=0, tol=1e-7)
top_roll = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=x1-tol && x[0]<=x2+tol", side=H_gap/2, x1=-l_r, x2=0, tol=1e-7)
top_up = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]<=x1+tol", side=H_gap/2, x1=-l_r, tol=1e-7)
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
gam = TestFunction(V_t)
T = TrialFunction(V_t)
T_ = Function(V_t, name="temp")
V_a = FunctionSpace(mesh, "CG", 1)
beta = TestFunction(V_a)
dal = TrialFunction(V_a)
al = Function(V_a, name="alpha")
al_last = Function(V_a) # alpha at previous iteration
al_diff = Function(V_a) # difference in alpha from previous iteration

# Boundary conditions
T_roller = Expression("T", degree=1, T=T0)
bc_roller = DirichletBC(V_t, T_roller, facets, 5) # temperature at roller
T_in = Expression("T", degree=1, T=T0)
bc_T_in = DirichletBC(V_t, T_in, facets, 1) # initial temperature of tow
bcs_T = [bc_roller, bc_T_in]

alpha_in = Expression("a", degree=1, a=alpha0)
bc_a_in = DirichletBC(V_a, alpha_in, facets, 1) # initial degree of cure of tow
bcs_a = [bc_a_in]

# Initial conditions
T_init = Expression("T", degree=1, T=T0)
T_.assign(project(T_init, V_t))
alpha_init = Expression("a", degree=1, a=alpha0)
al.assign(project(alpha_init, V_a))
al_last.assign(project(alpha_init, V_a))

# Parameters for numerical model
stablz = CellDiameter(mesh)
x = MeshCoordinates(mesh)
n = FacetNormal(mesh)

v_ = Expression(("v","0."), degree=1, v=V_r) # Velocity field
# phi = Expression() # fiber volume fraction field
class FiberVolFrac(UserExpression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def eval(self,values,x):
        if x[0] <= -l_r:
            phi = phi_1
        elif x[0] <= 0:
            phi = H_tow*phi_1 / (H_tow - 2*(sqrt(R_r**2 - x[0]**2) - R_r*cos(a_r)))
        else:
            phi = phi_2
        values[0] = phi

    def value_shape(self):
        return ()

phi = FiberVolFrac()

# Homogenized properties
k_par = kr*(1-phi) + kf*phi # thermal cond. parallel to fibers (x axis)
k_perp = kr + 2*phi*kr / ((kf+kr)/(kf-kr) - phi + (kf-kr)/(kf+kr)*(0.30584*phi**4 + 0.013363*phi**8)) # thermal cond. perpendiculat to fibers (y axis)
k_tens = as_tensor([[k_par, 0],[0, k_perp]]) # thermal conductivity tensor
rho_bar = rhor*(1-phi) + rhof*phi # homogenized density
Cp_bar = (Cpr*rhor*(1-phi) + Cpf*rhof*phi)/rho_bar # homogenized specific heat
rho_Cp_bar = Cpr*rhor*(1-phi) + Cpf*rhof*phi # product of homogenized density and specific heat

# Variational form for degree of cure problem
tau_a = stablz
g = (1-al)**n_ * al**m_ / (1+exp(Ca*(al-alpha_c)))
F_a = (beta + tau_a*dot(v_, grad(beta)))*(dot(v_, grad(al)) - A_*exp(-Er/R_/T_)*g)*dx
tang_a = derivative(F_a, al, dal)

# Variational form for thermal problem
Pe = V_r*stablz/2/(k_par/rho_Cp_bar) # Peclet number for thermal problem (diffusivity in x direction)
tau_T = stablz * (1/tanh_ufl(Pe) - 1/Pe)
# ONLY APPLYING SUPG TEST FN TO MATERIAL DERIVATIVE TERMS
F_T = dot(grad(gam),dot(k_tens, grad(T)))*dx + (gam + tau_T*dot(v_, grad(gam)))*(rho_Cp_bar*dot(v_, grad(T)) \
                                                        - rhor*Hr*(1-phi)*dot(v_, grad(al)))*dx + gam*h_conv*(T - Tamb)*ds(4)

# Solver for degree of cure
# problem_alpha = LinearVariationalProblem(lhs(F_a), rhs(F_a), al_, bcs_a, form_compiler_parameters=ffc_options)
# solver_alpha = LinearVariationalSolver(problem_alpha)
problem_alpha = NonlinearVariationalProblem(F_a, al, bcs=bcs_a, J=tang_a, form_compiler_parameters=ffc_options)
solver_alpha = NonlinearVariationalSolver(problem_alpha)
prms_a = solver_alpha.parameters
prms_a["nonlinear_solver"] = "newton"
prms_a["newton_solver"]["linear_solver"] = "default"
prms_a["newton_solver"]["report"] = True
prms_a["newton_solver"]["absolute_tolerance"] = 1e-19
prms_a["newton_solver"]["relative_tolerance"] = 1e-19
prms_a["newton_solver"]["error_on_nonconvergence"] = False
prms_a["newton_solver"]["convergence_criterion"] = "residual"
prms_a["newton_solver"]["maximum_iterations"] = 20
prms_a["newton_solver"]["relaxation_parameter"] = 1.

# Solver for temperature
problem_temp = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs_T, form_compiler_parameters=ffc_options)
solver_temp = LinearVariationalSolver(problem_temp)

################################### SOLUTION #################################################

# Output file
file_results = XDMFFile("results_comp_print_roller.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

file_results.write(T_, 0)
file_results.write(al, 0)

# Write functions for parameters that vary with fiber volume fraction
V_tens = TensorFunctionSpace(mesh, "CG", 1)
k_fun = Function(V_tens, name="therm_cond")
k_fun.assign(project(k_tens, V_tens))
file_results.write(k_fun, 0)

phi_fun = Function(V_t, name="fiber_vol_frac")
phi_fun.assign(project(phi, V_t))
file_results.write(phi_fun, 0)

rho_Cp_fun = Function(V_t, name="rho_Cp_bar")
rho_Cp_fun.assign(project(rho_Cp_bar, V_t))
file_results.write(rho_Cp_fun, 0)

# Get dofs for midline
alpha_coords = V_a.tabulate_dof_coordinates()
alpha_dof = np.array(V_a.dofmap().dofs()) # dofs for alpha among all processors
alpha_midline_dofs_local = np.where(alpha_coords[:,1] <= 0)[0] # midline dofs, using this core's dofs
alpha_midline_dofs = alpha_dof[alpha_midline_dofs_local] # midline dofs, using global dofs

T_roller_list = []
alpha_max_list = []
front_loc_list = []


load_prev = 0
# Iterative solution for steady problem
for l_iter, load in enumerate(loads):
    
    print_parallel("Solving for load {}".format(load))

    # Update bounday condition for load
    T_roller.T = T0 + load*(T_r - T0) # ramp roller temperature

    for k in range(1, max_iter+1):
        solver_temp.solve()
        (it_nl, conv_nl) = solver_alpha.solve()

        # If solution for alpha does not converge, use alternative solver
        if conv_nl != 1:
            print("Not converged")
            
            al.assign(al_last)

            x_ = Function(V_a) # solution vector 
            tol_abs_nl = 1e-19
            tol_inc_nl = 1e-13
            max_iter_nl = 20
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
            error_inc = 1

            while error > tol_abs_nl and error_inc > tol_inc_nl and it <= max_iter_nl:
                it += 1
                A, b = assemble_system(tang_a, -F_a, bcs_a_hold)
                solve(A, x_.vector(), b)
                error = b.norm('l2')
                error_inc = norm(x_)
                print_parallel('Iteration: {:d}; Residual: {:.3e}; Incremental: {:.3e} '.format(it, error, error_inc))
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
        # print_parallel(x_coord_sorted)
        # print_parallel(alpha_sorted)
    x_midline = comm.bcast(x_coord_sorted, root=0)
    alpha_midline = comm.bcast(alpha_sorted, root=0)

    alpha_max = max(alpha_midline)
    try:
        front_loc = interp1d(alpha_midline, x_midline)(0.5)
    except:
        front_loc = -np.inf

    T_roller_list.append(T0 + load*(T_r - T0))
    alpha_max_list.append(alpha_max)
    front_loc_list.append(front_loc)
    
    
    # Write converged solution for load step
    file_results.write(T_, load)
    file_results.write(al, load)

    
    load_prev = load

print_parallel("")
print_parallel("Load no\tT_r [degC]\talpha_max\tfront_x [mm]")
for i, (T_r_i, alpha_max_i, front_loc_i) in enumerate(zip(T_roller_list, alpha_max_list, front_loc_list)):
    print_parallel("{}\t{}\t{}\t{}".format(i, T_r_i-273.15, alpha_max_i, front_loc_i*1e3))
