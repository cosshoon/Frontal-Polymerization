# Michael Zakoworotny, Gavin DeBrun
# 1D DCPD thermo-chemical model, with uniform advective velocity

from dolfin import *
from mshr import * # for making simple geometries
import os
import sys
import time
from math import floor, pi, tanh, isnan
import numpy as np
from ufl import tanh as tanh_ufl
from scipy.interpolate import interp1d
# from mpi4py import MPI

####################### SETUP SOLVER ENVIRONMENT ####################

set_log_level(30)

comm = MPI.comm_world #MPI.COMM_WORLD #
comm_rank = MPI.rank(comm) #comm.Get_rank() #

# Handling paths
try:
    os.chdir(os.path.dirname(__file__))
except:
    if comm_rank==0:
        print("Cannot switch directories")

parameters["ghost_mode"]="shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

############################## PARAMETERS #####################################
# Thermo-chemical properties
kr =  0.15
rhor = 980.0 
Cpr = 1600.0
R_ = 8.314

# Nature values
Hr =  350000
A_ = 8.55e15
Er = 110750.0
n = 1.72
m = 0.77
Ca = 14.48
alpha_c = 0.41
# # Aditya fit n-th order
# Hr = 360000
# A_ = 1.5385e12
# Er = 92799.0
# n = 2.4


# Geometry - simple bar (rectangle)
L = 0.025 #0.015
h_x = 2e-5#5e-6
nel_x = round(L/h_x) #1000
R = 0.00077 #0.001174 #0.00154/2

# Convection
h_conv = 50
T_amb = 60 + 273.15

v = 1.6e-3 #1.1e-3 #0.841e-3 # convective velocity

# Time stepping parameters
tend = 140#10
dt = 0.001
num_steps = int(tend/dt)

# Output frequency
out_freq = 0.2
out_step = round(out_freq/dt)

# Initial conditions
alpha0 = 0.1
T0 = 20+273.15
# Boundary conditions
T_trig = T0 + Hr/Cpr*(1-alpha0)#180+273.15
time_trig = 20.0

####################### PRE-PROCESSING ##########################

# Create mesh and define function space
mesh = IntervalMesh(nel_x,0.0,L)

# Set boundaries
left =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 0.0, tol=1e-7)
right =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = L, tol=1e-7)
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
facets.set_all(0)
left.mark(facets, 1)
right.mark(facets, 2)
ds = Measure('ds', domain=mesh, subdomain_data=facets) # partition boundary

# Set function space
V = FunctionSpace(mesh, "CG", 1) 
# Define functions
w = TestFunction(V)
T = TrialFunction(V)
T_n = Function(V)
T_ = Function(V, name="Temp")
T_max = Function(V, name="Max_temp")

U = FunctionSpace(mesh, "CG", 1) 
# Define functions
beta = TestFunction(U)
alpha = TrialFunction(U)
alpha_n = Function(U)
alpha_ = Function(U, name="alpha")

# Boundary conditions
bc_inlet = DirichletBC(V, Constant(T0), facets, 1)
T_trig_val = Constant(T_trig)
bc_trig = DirichletBC(V, T_trig_val, facets, 2)
bcs_T = [bc_inlet, bc_trig]

bcs_a = [DirichletBC(U, Constant(alpha0), facets, 1)]

# Initial values
T_.interpolate(Constant(T0))
T_n.interpolate(Constant(T0))
alpha_.interpolate(Constant(alpha0))
alpha_n.interpolate(Constant(alpha0))

# Define variational problem
h = CellDiameter(mesh)
stablz = h/2/v
# g = (1-alpha)**n
# g = (1-alpha_n)**n   # max_value((1-al), 0)**n_   #(1-al)**n_ * al**m_ / (1+exp(Ca*(al-alpha_c)))
g = (1-alpha_n)**n * alpha_n**m / (1+exp(Ca*(alpha_n-alpha_c)))
tau_a = stablz
F_a =  (beta + tau_a*v*beta.dx(0))*((alpha - alpha_n)/dt + v*alpha.dx(0) - A_*exp(-Er/R_/T_n)*g)*dx
# Implicit, alpha, implicit temperature
# F_T = (dt*kr*dot(grad(w),grad(T)) + rhor*Cpr*w*T - rhor*Cpr*w*T_n - rhor*Hr*w*alpha_ + rhor*Hr*w*alpha_n)*dx
Pe = v*h/2/(kr/Cpr/rhor) # Peclet number for thermal problem
tau_T = stablz * (1/tanh_ufl(Pe) - 1/Pe)
q_conv = 2/R*h_conv*(T - T_amb)
F_T = (kr*w.dx(0)*T.dx(0) + (w + tau_T*v*w.dx(0))*(rhor*Cpr*((T - T_n)/dt + v*T.dx(0)) \
                                - rhor*Hr*((alpha_ - alpha_n)/dt + v*alpha_.dx(0))))*dx + w*q_conv*dx

# F_a = (beta*alpha - beta*alpha_n - dt*A_*beta*exp(-Er/R_/T_)*g)*dx

# tang = derivative(F_a, alpha, dalpha)

problem_a = LinearVariationalProblem(lhs(F_a), rhs(F_a), alpha_, bcs=bcs_a, form_compiler_parameters=ffc_options)
solver_a = LinearVariationalSolver(problem_a)

prms_a = solver_a.parameters
prms_a['linear_solver'] = 'gmres'
# prms_a['preconditioner'] = 'hypre_amg'

problem_T = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T, form_compiler_parameters=ffc_options)
solver_T = LinearVariationalSolver(problem_T)

prms_T = solver_T.parameters
prms_T['linear_solver'] = 'gmres'
# prms_T['preconditioner'] = 'hypre_amg'


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
file_results = XDMFFile("results_1D_DCPD_decoupled.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

t_array = np.zeros((num_steps+1,1))
pos_array = np.full((num_steps+1,1), L)#np.zeros((num_steps+1,1))
# x_array = np.linspace(0,L,num=nel_x+1)

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
    
    # Remove trigger
    if t >= time_trig and not stop_trigger:
        bcs_T = [bc_inlet]
        problem_T = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T, form_compiler_parameters=ffc_options)
        solver_T = LinearVariationalSolver(problem_T)
        stop_trigger = True

    # Solve alpha explicitly, T implicitly
    # local_project(alpha_n + dt*A_*exp(-Er/(R_*T_n))*g, Q, alpha_)
    solver_a.solve()
    alpha_.vector()[:] = np.minimum(alpha_.vector()[:],0.999)
    alpha_.vector()[:] = np.maximum(alpha_.vector()[:],alpha0-1e-4)
    solver_T.solve()

    # Track maximum temperature over time
    T_max.vector()[:] = np.maximum(T_max.vector()[:], T_.vector()[:])

    # Find location of front for velocity computation (only for serial - may not work as intended in parallel)
    t_array[step] = t
    if max(alpha_.vector()[:]) >= 0.5 and min(alpha_.vector()[:]) <= 0.5:
        # pos_array[step] = U.tabulate_dof_coordinates()[np.argmin(abs(alpha_.vector()[:] - 0.5))] #np.where(abs(alpha_.vector()[:] - 0.5) == min(abs(alpha_.vector()[:] - 0.5)) )[0][0]]
        pos_array[step] = interp1d(alpha_.vector()[:], alpha_.function_space().tabulate_dof_coordinates()[:,0])(0.5)

    T_n.assign(T_)
    alpha_n.assign(alpha_)
    
    # Save to file 
    if step % out_step == 0:
        if comm_rank == 0:
            print("Saving to results file at t=%f" % t)
            sys.stdout.flush()
        file_results.write(T_, t)
        file_results.write(alpha_, t)
        file_results.write(T_max, t)

    step += 1

np.savetxt("front_pos.csv", np.hstack((t_array, pos_array)), delimiter=",")
fit = np.polyfit(t_array[np.logical_and(pos_array>0.1*L, pos_array<0.9*L)], pos_array[np.logical_and(pos_array>0.1*L, pos_array<0.9*L)], 1)
vel_avg = fit[0]
print(vel_avg)
