# Michael Zakoworotny, Gavin DeBrun
# Modeling the thermo-chemical behavior of impregnated carbon fiber tow as it 
# is deposited on a heated substrate


from dolfin import *
import os
import sys
import time
from math import ceil, floor, pi, tanh
import numpy as np
import gmsh
import meshio
from mpi4py import MPI
import csv

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

# Print message on only one process
def print_parallel(mssg):
    if comm_rank==0:
        print(mssg)
        sys.stdout.flush()

set_log_level(30) # error level=40, warning level=30, info level=20
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

# the following needs to be added to use quadrature elements
parameters["form_compiler"]["representation"] = 'quadrature'  #this is deprecated
# The following shuts off a deprecation warning for quadrature representation:
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

dx = dx(degree=QUAD_DEG, scheme="default")

#For local projections
def local_project(v, V, u):
    parameters["form_compiler"]["representation"] = 'quadrature'
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    solver.solve_local_rhs(u)


######################## PARAMETERS ########################

# Geometry
L = 50e-3 # total length of tow is 100 mm
h = 0.64e-3 # height of the semi-elliptical region tow that is deposited on the substrate

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

# # Carbon fiber properties (from Vyas 2020)
# rhof = 1800.0
# kf = 9.363
# Cpf = 753.6

# HexTow AS4 Carbon Fiber
rhof = 1790.0
kf = 6.83
Cpf = 1129.0

# Fiber volume fraction
phi = 0.27 # experimental value from Nadim's experiment

# Composite properties
rho_Cp_bar = rhor*Cpr*(1-phi) + rhof*Cpf*phi # homogenized density times homogenized Cp
k_par = kr*(1-phi) + kf*phi
k_perp = kr*kf / ((1-phi)*kf + phi*kr)

# Process parameters
Vprint = 5e-3
Tsub = 100 + 273.15

# Environmental conditions (convection)
h_conv = 50
Tamb = 30 + 273.15

# Initial conditions
T0 = 20 + 273.15
alpha0 = 0.1

# Time stepping
tend = L/Vprint*0.99
dt = 1e-3 #5e-4#0.01#0.01
num_steps = int(tend/dt)

# Number of steps at which material is added to domain
add_time = 0.025
add_steps = round(add_time/dt)

# Output frequency
out_freq = 0.05
out_step = round(out_freq/dt)

######################## SETUP PROBLEM ###########################

h_elem = 2e-5
nel_x = ceil(L/h_elem)
nel_y = ceil(h/h_elem)
mesh = RectangleMesh(Point(0., 0.), Point(L, h), nel_x, nel_y)

x_printer = 0

# Define boundaries
left = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=0.0, tol=1e-7)
bot_printed = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]<=xp+tol", side=0.0, xp=x_printer, tol=1e-7)
bot_outside = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=xp-tol", side=0.0, xp=x_printer, tol=1e-7)
right = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=L, tol=1e-7)
top_outside = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=xp-tol", side=h, xp=x_printer, tol=1e-7)
top_printed = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]<=xp+tol", side=h, xp=x_printer, tol=1e-7)
# Mark boundaries
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
facets.set_all(0)
left.mark(facets, 1)
bot_printed.mark(facets, 2)
bot_outside.mark(facets, 3)
right.mark(facets, 4)
top_outside.mark(facets, 5)
top_printed.mark(facets, 6)
# ds = Measure('ds', domain=mesh, subdomain_data=facets) # partition boundary
ds = ds(subdomain_data=facets)

# Define domains
printed = CompiledSubDomain("x[0] <= xp", xp=x_printer) 
# Mark domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
printed.mark(domains, 1)
dx = dx(subdomain_data = domains)


# Function spaces
V = FunctionSpace(mesh, "CG", 1) # temperature
Qelement = FiniteElement("Quadrature", mesh.ufl_cell(), degree=QUAD_DEG, quad_scheme="default")
Q = FunctionSpace(mesh, Qelement) # alpha

# Define functions
w = TestFunction(V)
T = TrialFunction(V)
T_n = Function(V)
T_ = Function(V, name="Temp")
T_max = Function(V, name="Max_temp")

alpha_ = Function(Q)    # function to store current value of alpha
alpha_n = Function(Q)   # function to store previous value of alpha

# Define an additional space for outputing alpha to results file
Y2 = FunctionSpace(mesh, "DG", 0) # Use piecewise constant interpolation to prevent overshooting at nodes (ie. alpha > 1)
alpha_r = Function(Y2, name="alpha")

# Boundary conditions
T_substrate = Constant(Tsub)
bc_substrate = DirichletBC(V, T_substrate, facets, 2) # temperature of substrate
bcs_T = [bc_substrate]

# Initial conditions
T_.interpolate(Constant(T0))
T_n.interpolate(Constant(T0))
alpha_n.interpolate(Constant(alpha0))

# Define variational problem
g = (1-alpha_n)**n_ * alpha_n**m_ / (1+exp(Ca*(alpha_n-alpha_c))) # g(alpha) for PTD model

k_tens = as_tensor([[k_par, 0],[0, k_perp]])

F_T = dot(grad(w),dot(k_tens, grad(T)))*dx + rho_Cp_bar*w*(T - T_n)/dt*dx - rhor*Hr*(1-phi)*w*(alpha_ - alpha_n)/dt*dx

problem_T = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T, form_compiler_parameters=ffc_options)
solver_T = LinearVariationalSolver(problem_T)


################################### SOLUTION #################################################

# Output file
file_results = XDMFFile("results_2D_DCPD.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Time-stepping
t = 0
step=1
start_time = time.time()
stop_trigger = False
while step < num_steps+1:
    
    t += dt

    if comm_rank==0:
        print("t = %f, time elapsed = %f" % (t, time.time()-start_time))
        sys.stdout.flush()

    x_printer += dt*Vprint # move printer coordinate

    if step % add_steps == 0:
        # Update boundaries
        bot_printed = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]<=xp+tol", side=0.0, xp=x_printer, tol=1e-7)
        bot_outside = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=xp-tol", side=0.0, xp=x_printer, tol=1e-7)
        top_outside = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=xp-tol", side=h, xp=x_printer, tol=1e-7)
        top_printed = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]<=xp+tol", side=h, xp=x_printer, tol=1e-7)
        bot_printed.mark(facets, 2)
        bot_outside.mark(facets, 3)
        top_outside.mark(facets, 5)
        top_printed.mark(facets, 6)
        ds = ds(subdomain_data=facets)

        # Update domains
        printed = CompiledSubDomain("x[0] <= xp", xp=x_printer) 
        printed.mark(domains, 1)
        dx = dx(subdomain_data = domains)

        # Rebuild problem
        bc_substrate = DirichletBC(V, T_substrate, facets, 2) # temperature of substrate
        bcs_T = [bc_substrate]
        problem_T = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T, form_compiler_parameters=ffc_options)
        solver_T = LinearVariationalSolver(problem_T)


    local_project(alpha_n + dt*A_*exp(-Er/(R_*T_n))*g, Q, alpha_)
    alpha_.vector()[:] = np.minimum(alpha_.vector()[:],0.999)
    alpha_.vector()[:] = np.maximum(alpha_.vector()[:],alpha0-1e-4)
    solver_T.solve()

    T_n.assign(T_)
    alpha_n.assign(alpha_)

    if step % out_step == 0:
        print_parallel("Saving to results file at t=%f" % t)

        local_project(alpha_, Y2, alpha_r)
        file_results.write(T_, t)
        file_results.write(alpha_r, t)


    step += 1