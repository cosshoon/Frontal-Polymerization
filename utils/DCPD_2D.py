# Michael Zakoworotny, Gavin DeBrun
# 2D DCPD thermo-chemical model

from dolfin import *
import os
import sys
import time
import numpy as np
import math
# from mpi4py import MPI

####################### SETUP SOLVER ENVIRONMENT ####################

set_log_level(30) # turn off informational messages from solver

comm = MPI.comm_world # get the MPI communicator for parallelization
comm_rank = MPI.rank(comm) # rank of comm (ie. number of process)

def print_parallel(mssg):
    if comm_rank==0:
        print(mssg)
        sys.stdout.flush()

# Handling paths - switch to directory of file
try:
    os.chdir(os.path.dirname(__file__))
except:
    print_parallel("Cannot switch directories")

# Settings for fenics
parameters["ghost_mode"]="shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Use quadrature representation when using quadrature elements
parameters["form_compiler"]["representation"] = 'quadrature'  #this is deprecated
# The following shuts off a deprecation warning for quadrature representation:
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

############################## PARAMETERS #####################################
## Thermal properties ##
kr =  0.15          # thermal conductivity, W/m-K
rho = 980.0         # density, kg/m^3
Cp = 1600.0         # specific heat, J/kg-K
R_ = 8.314          # gas constant, J/mol-K

## Cure kinetics parameters ##

# Values from Nature paper, PTD model (Robertson, et al., 2018)
Hr =  350000      # Enthalpy, J/kg
A_ = 8.55e15      # time constant, 1/s
Er = 110750.0     # activation energy, J/mol
n = 1.72          # constant for g(alpha)
m = 0.77          # constant for g(alpha)
Ca = 14.48        # constant for g(alpha)
alpha_c = 0.41    # constant for g(alpha)

# # Aditya fit to n-th order model
# Hr = 360000         # Enthalpy, J/kg
# A_ = 1.5385e12      # time constant, 1/s
# Er = 92799.0        # activation energy, J/mol
# n = 2.4             # constant for g(alpha)

## Geometry - 1D bar ##
L = 0.005             # x dimension (m)
H = 0.00154/2           # y dimension (m)
h_el = 1.25e-5         # element size
nel_x = round(L/h_el)  # number of x elements
nel_y = round(H/h_el)  # number of x elements

## Time stepping parameters ##
tend = 10           # end time of simulation (s)
dt = 0.001          # fixed time step size (s)
num_steps = int(tend/dt)    # number of steps in simulation

## Output frequency ##
out_freq = 0.1      # frequency at which to output results to file (s)
out_step = round(out_freq/dt)   # number of steps between outputs

## Initial conditions ##
alpha0 = 0.02       # initial degree of cure
T0 = 20+273.15      # initial temperature (K)

## Boundary conditions ##
T_trig = 180+273.15 # trigger temperature (K)
time_trig = 1.0     # duration of trigger (s)


####################### PRE-PROCESSING ##########################

# Create 2D mesh
mesh = RectangleMesh(Point(0.,0.), Point(L, H), nel_x, nel_y, "left/right") # "left/right" for alternating tri element direction

# Define boundaries of domain
left = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=0.0, tol=1e-7)
bot = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=0.0, tol=1e-7)
right = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=L, tol=1e-7)
top = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=H, tol=1e-7)
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
facets.set_all(0)
left.mark(facets, 1)    # mark left boundary with label 1
bot.mark(facets, 2)   # mark bottom boundary with label 2
right.mark(facets, 3)    # mark right boundary with label 3
top.mark(facets, 4)   # mark top boundary with label 4
ds = Measure('ds', domain=mesh, subdomain_data=facets) # redefine the measure ds with markings.
                                                       # Can integrate over marked boundary with ds(1), ds(2), etc

# Define function space for temperature - 1st order, continuous Lagrange elements
V = FunctionSpace(mesh, "CG", 1) 
# Define functions
w = TestFunction(V)     # test function for temperature
T = TrialFunction(V)    # trial function for temperature
T_ = Function(V, name="Temp")   # function to store the current value of temperature, and for post-processing
T_n = Function(V)               # function to store the previous value of temperature
T_max = Function(V, name="Max_temp")    # function to track maximum value of temp at each node over time

# Setup quadrature rule
QUAD_DEG = 4    # order of quadrature rule
dx = dx(degree=QUAD_DEG, scheme="default")    # redefine area integral to use same quadrature order
parameters["form_compiler"]["quadrature_degree"] = QUAD_DEG
# Define quadrature function space for degree of cure - DOFs stored only at quadrature points, no shape functions
Qelement = FiniteElement("Quadrature", mesh.ufl_cell(), degree=QUAD_DEG, quad_scheme="default")
Q = FunctionSpace(mesh, Qelement)
# Define functions
alpha_ = Function(Q)    # function to store current value of alpha
alpha_n = Function(Q)   # function to store previous value of alpha
# Define an additional space for outputing alpha to results file
Y2 = FunctionSpace(mesh, "DG", 0) # Use piecewise constant interpolation to prevent overshooting at nodes (ie. alpha > 1)
alpha_r = Function(Y2, name="alpha")

# Boundary conditions for temperature
T_trig_val = Constant(T_trig)   # fenics wrapper for a constant
bc_trig = DirichletBC(V, T_trig_val, facets, 1)     # set temperature to T_trig on left boundary
bcs_T = [bc_trig]   # list of temperature boundary conditions

# Initial values
T_.interpolate(Constant(T0))    # set initial value of T_ to T0
T_n.interpolate(Constant(T0))   # set initial value of T_n to 
alpha_n.interpolate(Constant(alpha0))   # set initial value of alpha_n to alpha_
# Don't need to set alpha_ because it will be immedately overwritten

# Define variational problem
g = (1-alpha_n)**n * alpha_n**m / (1+exp(Ca*(alpha_n-alpha_c))) # g(alpha) for PTD model
# g = (1-alpha_n)**n                                              # g(alpha) for nth order model
# Implicit, alpha, implicit temperature
F_T = (kr*dot(grad(w),grad(T)) + rho*Cp*w*(T - T_n)/dt - rho*Hr*w*(alpha_ - alpha_n)/dt)*dx
# NOTE: to include boundary integrals, use the function ds to refer to a differential element on the domain surface
# F_a = (beta*alpha - beta*alpha_n - dt*A_*beta*exp(-Er/R_/T_)*g)*dx

# tang = derivative(F_a, alpha, dalpha)

problem_T = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T, form_compiler_parameters=ffc_options)
solver_T = LinearVariationalSolver(problem_T)


def local_project(v, V, u):
    parameters["form_compiler"]["representation"] = 'quadrature'
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    solver.solve_local_rhs(u)

####################### SOLUTION LOOPS ##########################

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
    
    if t >= time_trig and not stop_trigger:
        bcs_T = []
        problem_T = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T, form_compiler_parameters=ffc_options)
        solver_T = LinearVariationalSolver(problem_T)
        stop_trigger = True

    local_project(alpha_n + dt*A_*exp(-Er/(R_*T_n))*g, Q, alpha_)
    alpha_.vector()[:] = np.minimum(alpha_.vector()[:],0.999)
    alpha_.vector()[:] = np.maximum(alpha_.vector()[:],alpha0-1e-4)
    solver_T.solve()

    T_max.vector()[:] = np.maximum(T_max.vector()[:], T_.vector()[:])

    T_n.assign(T_)
    alpha_n.assign(alpha_)
    
    # Save to file 
    if step % out_step == 0:
        local_project(alpha_, Y2, alpha_r)
        if comm_rank == 0:
            print("Saving to results file at t=%f" % t)
            sys.stdout.flush()
        file_results.write(T_, t)
        file_results.write(alpha_r, t)
        file_results.write(T_max, t)

    step += 1
