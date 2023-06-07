# Michael Zakoworotny, Gavin DeBrun
# Modeling the thermo-chemical behavior of a carbon-fiber reinforced
# laminate composite, using a 3D model that is homogenized in each lamina

from dolfin import *
import os
import sys
import time
from math import ceil, floor, pi, tanh
import numpy as np
from mpi4py import MPI
import csv
from scipy.interpolate import interp1d

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()


# Print message on only one process
def print_parallel(mssg):
    if comm_rank == 0:
        print(mssg)
        sys.stdout.flush()


set_log_level(30)  # error level=40, warning level=30, info level=20
# Switch to directory of file
try:
    os.chdir(os.path.dirname(__file__))
except:
    print_parallel("Cannot switch directories")

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True,
}

QUAD_DEG = 2
parameters["form_compiler"]["quadrature_degree"] = QUAD_DEG

# the following needs to be added to use quadrature elements
parameters["form_compiler"]["representation"] = "quadrature"  # this is deprecated
# The following shuts off a deprecation warning for quadrature representation:
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

dx = dx(degree=QUAD_DEG, scheme="default")


# For local projections
def local_project(v, V, u):
    parameters["form_compiler"]["representation"] = "quadrature"
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_) * dx
    b_proj = inner(v, v_) * dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    solver.solve_local_rhs(u)


######################## PARAMETERS ########################

# Geometry
L = 20e-3  # length of laminate plate
wid = 2e-3  # width of laminate plate
# t_lam = 1e-3 # thickness of each lamina

# Thermo-chemical parameters
rhor = 980
kr = 0.15
Cpr = 1600.0
R_ = 8.314
# Nature values
Hr = 350000
A_ = 8.55e15
Er = 110750.0
n_ = 1.72
m_ = 0.77
Ca = 14.48
alpha_c = 0.41

# HexTow AS4 Carbon Fiber
rhof = 1790.0
kf1 = 6.83
kf2 = 2.7
Cpf = 1129.0

# Fiber volume fraction
phi = 0.3
# Fiber angles
beta_list = np.array([-45,45])
# Ply thickness list
t_list = np.array([1e-3, 1e-3])  # length N
twoH = sum(t_list)

# Homogenized properties - as functions of input phi
k_1_fun = lambda p: kr * (1 - p) + kf1 * p  # thermal cond. parallel to fibers (x axis)

k_2_fun = lambda p: kr * (kf2 * (1 + p) + kr * (1 - p)) / (kf2 * (1 - p) + kr * (1 + p))

# thermal cond. perpendicular to fibers (y axis)
rho_bar_fun = lambda p: rhor * (1 - p) + rhof * p  # homogenized density
Cp_bar_fun = lambda p: (Cpr * rhor * (1 - p) + Cpf * rhof * p) / rho_bar_fun(p)

# homogenized specific heat - product of homogenized density and specific heat
rho_Cp_bar_fun = lambda p: Cpr * rhor * (1 - p) + Cpf * rhof * p

# Initial conditions
T0 = 20 + 273.15
alpha0 = 0.1

# Trigger temperature
T_max = T0 + (1-phi)*rhor*Hr/rho_Cp_bar_fun(phi)*(1-alpha0)
T_trig = T_max + 20
time_trig = 8

# Time stepping
tend = 60
dt = 1e-3
num_steps = int(tend / dt)

# Output frequency
out_freq = 0.4
out_step = round(out_freq / dt)

######################## SETUP PROBLEM ###########################

h_elem_x = 2e-5
h_elem_y = 4e-5
h_elem_z = 4e-5
nel_x = ceil(L / h_elem_x)
nel_y = ceil(wid / h_elem_y)
nel_z = ceil(twoH / h_elem_z)
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(L, wid, twoH), nel_x, nel_y, nel_z)

# Define boundaries
left = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=0.0, tol=1e-7)
bot = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=0.0, tol=1e-7)
back = CompiledSubDomain("on_boundary && near(x[2], side, tol)", side=0.0, tol=1e-7)
right = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=L, tol=1e-7)
top = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=wid, tol=1e-7)
front = CompiledSubDomain("on_boundary && near(x[2], side, tol)", side=twoH, tol=1e-7)
# Mark boundaries
facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facets.set_all(0)
left.mark(facets, 1)
bot.mark(facets, 2)
back.mark(facets, 3)
right.mark(facets, 4)
top.mark(facets, 5)
front.mark(facets, 6)
ds = ds(subdomain_data=facets)

# Function spaces
V = FunctionSpace(mesh, "CG", 1)  # temperature
Qelement = FiniteElement("Quadrature", mesh.ufl_cell(), degree=QUAD_DEG, quad_scheme="default")
Q = FunctionSpace(mesh, Qelement)  # alpha

# Define functions
w = TestFunction(V)
T = TrialFunction(V)
T_n = Function(V)
T_ = Function(V, name="Temp")
T_max = Function(V, name="Max_temp")

alpha_ = Function(Q)  # function to store current value of alpha
alpha_n = Function(Q)  # function to store previous value of alpha

# Define an additional space for outputing alpha to results file
# Y2 = FunctionSpace(mesh, "DG", 0)
Y2 = FunctionSpace(mesh, "CG", 1)
# Use piecewise constant interpolation to prevent overshooting at nodes (ie. alpha > 1)
alpha_r = Function(Y2, name="alpha")

# Boundary conditions
T_trig_val = Constant(T_trig)
bc_trig = DirichletBC(V, T_trig_val, facets, 1)  # temperature of trigger
bcs_T_trig = [bc_trig]
bcs_T_adiab = []

# Initial conditions
T_.interpolate(Constant(T0))
T_n.interpolate(Constant(T0))
alpha_n.interpolate(Constant(alpha0))
alpha_r.interpolate(Constant(alpha0))

# Define variational problem
g = (1 - alpha_n) ** n_ * alpha_n**m_ / (1 + exp(Ca * (alpha_n - alpha_c)))
# g(alpha) for PTD model

# # Thermal conductivity matrix for laminate (in 3D)
# class ThermalCond(UserExpression):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         self.bnds = np.insert(np.cumsum(t_list), 0, 0)
         
#     def eval(self,values,x):
#         z = x[2]
#         ind = np.where(np.logical_and(z <= self.bnds[1:], z >= self.bnds[:-1]))[0][0]
#         beta_cur = np.radians(beta_list[ind])
#         kxx = k_1_fun(phi) * np.cos(beta_cur) ** 2 + k_2_fun(phi) * np.sin(beta_cur) ** 2
#         kxy = np.sin(beta_cur) * np.cos(beta_cur) * (k_1_fun(phi) - k_2_fun(phi))
#         kyy = k_1_fun(phi) * np.sin(beta_cur) ** 2 + k_2_fun(phi) * np.cos(beta_cur) ** 2
#         kzz = k_2_fun(phi)
#         k = np.array([[kxx, kxy, 0], [kxy, kyy, 0], [0, 0, kzz]])
#         values[:] = k.flatten()

#     def value_shape(self):
#         return (3,3)

# k_tens = ThermalCond() # expression for thermal conductivity tensor

# Define a quadrature space for tensors
Qtelement = TensorElement("Quadrature", mesh.ufl_cell(), degree=QUAD_DEG, quad_scheme="default")
Qt = FunctionSpace(mesh, Qtelement)  # tensor function space defined at quadrature points
k_tens_fn = Function(Qt)

# Vectorized assignment to tensor elements
# start_time = time.time()
bnds = np.insert(np.cumsum(t_list), 0, 0)
dof_coords = Qt.tabulate_dof_coordinates() # coords of all DOF in space
all_dofs = Qt.dofmap().dofs()
for i in range(9):
    Qi_dofs_glob = np.array(Qt.sub(i).dofmap().dofs())
    Qi_dofs_loc = np.flatnonzero(np.isin(all_dofs, Qi_dofs_glob)) # <- these give the DOF number for this processor
    z = np.reshape(dof_coords[Qi_dofs_loc,2], (-1,1)) # z values for these dofs
    inds = np.argmax(np.logical_and(z <= bnds[1:], z >= bnds[:-1]),axis=1)
    beta_cur = np.radians(beta_list[inds])
    if i == 0: # kxx
        k_tens_fn.vector()[Qi_dofs_loc] = k_1_fun(phi)*np.cos(beta_cur)**2 + k_2_fun(phi)*np.sin(beta_cur)**2
    elif i == 1 or i == 3: # kxy
        k_tens_fn.vector()[Qi_dofs_loc] = np.sin(beta_cur)*np.cos(beta_cur)*(k_1_fun(phi) - k_2_fun(phi))
    elif i == 4: # kyy
        k_tens_fn.vector()[Qi_dofs_loc] = k_1_fun(phi)*np.sin(beta_cur)**2 + k_2_fun(phi)*np.cos(beta_cur)**2
    elif i == 8: # kzz
        k_tens_fn.vector()[Qi_dofs_loc] = k_2_fun(phi)
    else: # all other 0
        k_tens_fn.vector()[Qi_dofs_loc] = 0
    # print_parallel("Interpolate time {} = {}".format(i, time.time() - start_time))


# k_tens_fn.interpolate(k_tens)
V_tens = TensorFunctionSpace(mesh, "DG", 0)
k_tens_plot = Function(V_tens, name="k_tens")
# k_tens_plot.interpolate(k_tens_fn)

# Get homogenized properties
# Specific heat and density
rho_bar = rho_bar_fun(phi)  # homogenized density
Cp_bar = Cp_bar_fun(phi)  # homogenized specific heat
rho_Cp_bar = rho_Cp_bar_fun(phi)  # product of homogenized density and specific heat

F_T = dot(grad(w), dot(k_tens_fn, grad(T))) * dx + \
      rho_Cp_bar * w * (T - T_n) / dt * dx - \
      rhor * Hr * (1 - phi) * w * (alpha_ - alpha_n) / dt * dx

# Solver with trigger boundary condition
problem_T_trig = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T_trig, form_compiler_parameters=ffc_options)
solver_T_trig = LinearVariationalSolver(problem_T_trig)
prms_T_trig = solver_T_trig.parameters
prms_T_trig["linear_solver"] = "cg"
prms_T_trig["preconditioner"] = "hypre_amg"
prms_T_trig["krylov_solver"]["absolute_tolerance"] = 1e-12
prms_T_trig["krylov_solver"]["relative_tolerance"] = 1e-12
prms_T_trig["krylov_solver"]["maximum_iterations"] = 100
# Solver with adiabatic boundary conditions
problem_T_adiab = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_T_adiab, form_compiler_parameters=ffc_options)
solver_T_adiab = LinearVariationalSolver(problem_T_adiab)
prms_T_adiab = solver_T_adiab.parameters
prms_T_adiab["linear_solver"] = "cg"
prms_T_adiab["preconditioner"] = "hypre_amg"
prms_T_adiab["krylov_solver"]["absolute_tolerance"] = 1e-12
prms_T_adiab["krylov_solver"]["relative_tolerance"] = 1e-12
prms_T_adiab["krylov_solver"]["maximum_iterations"] = 100

################################### SOLUTION #################################################

# Output file
file_results = XDMFFile("results_laminate_3D.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(T_, 0)
file_results.write(alpha_r, 0)
# start_time = time.time()
# local_project(k_tens_fn, V_tens, k_tens_plot)
# print_parallel("Local project time = {}".format(time.time() - start_time))
# file_results.write(k_tens_plot, 0)

# Determine lists of alpha dofs along horizontal lines at y values in front_y_list
# MAKE SURE THESE VALUE LIE ALONG NODAL POSITIONS
front_y_list = [0, wid/2, wid, 0, wid/2, wid, 0, wid/2, wid] # y coordinates of x lines along which to take alpha position
front_z_list = [0, 0, 0, twoH/2, twoH/2, twoH/2, twoH, twoH, twoH] # z coordinates of x lines
alpha_midline_dofs = [] # contains lists of the dof's corresponding to each line in lists(using global dofs among all processors)
alpha_midline_dofs_local = [] # using local dof on processor
alpha_coords = Y2.tabulate_dof_coordinates()
alpha_dof = np.array(V.dofmap().dofs())  # dofs for alpha among all processors
for i, (y, z) in enumerate(zip(front_y_list, front_z_list)):
    # dofs along this y value (with a tolerance), using this core's dofs
    alpha_midline_dofs_local.append(np.where(np.logical_and(np.isclose(alpha_coords[:,1],y,atol=1e-10), np.isclose(alpha_coords[:,2],z,atol=1e-10)))[0])
    # map these dofs to the corresponding global dofs among all cores, and add to list
    alpha_midline_dofs.append(alpha_dof[alpha_midline_dofs_local[i]])

# Initialize csv file
print_file = open("data_laminate_3D.csv",'w')
front_pos_header = ", ".join(["front_x at y/w={:.1f} z/2H={:.1f}".format(y/wid, z/twoH) for (y,z) in zip(front_y_list,front_z_list)])
alpha_max_header = ", ".join(["alpha_max at y/w={:.1f} z/2H={:.1f}".format(y/wid, z/twoH) for (y,z) in zip(front_y_list,front_z_list)])
print_file.write("t, {}, {}\n".format(front_pos_header, alpha_max_header))


# Time-stepping
t = 0
step = 1
start_time = time.time()
stop_trigger = False

# Solution loop
while step < num_steps + 1:
    t += dt

    print_parallel("t = %f, time elapsed = %f" % (t, time.time() - start_time))

    # Solver
    # First solve degree of cure explicitly
    local_project(alpha_n + dt * A_ * exp(-Er / (R_ * T_n)) * g, Q, alpha_)
    alpha_.vector()[:] = np.minimum(alpha_.vector()[:], 0.999)
    alpha_.vector()[:] = np.maximum(alpha_.vector()[:], alpha0 - 1e-4)

    if t < time_trig:
        solver_T_trig.solve()
    else:
        solver_T_adiab.solve()

    # Update for next time step
    T_n.assign(T_)
    alpha_n.assign(alpha_)

    # Get alpha profiles along each horizontal line, sorted from left to right
    # Then, compute the location where alpha is closest to 0.5
    front_loc = []
    alpha_max = []
    local_project(alpha_, Y2, alpha_r) # project alpha values to nodes
    # Iterate over each horizontal line
    for i, (y,z) in enumerate(zip(front_y_list,front_z_list)):
        x_coord_gather = comm.gather(alpha_coords[alpha_midline_dofs_local[i],0], root=0)
        alpha_gather = comm.gather(alpha_r.vector()[alpha_midline_dofs_local[i]], root=0)
        x_coord_sorted = []
        alpha_sorted = []
        if comm_rank == 0:
            x_coord_gathered = np.concatenate(x_coord_gather)
            alpha_gathered = np.concatenate(alpha_gather)
            x_coord_sort_inds = x_coord_gathered.argsort()
            x_coord_sorted = x_coord_gathered[x_coord_sort_inds]
            alpha_sorted = alpha_gathered[x_coord_sort_inds]
        x_line = comm.bcast(x_coord_sorted, root=0)
        alpha_line = comm.bcast(alpha_sorted, root=0)
        # Interpolate front position along this horizontal line
        try:
            front_loc.append(interp1d(alpha_line, x_line)(0.5))
        except:
            front_loc.append(0)
        # Maximum alpha value along this horizontal line
        alpha_max.append(max(alpha_line))


    # Write to XDMF
    if step % out_step == 0:
        print_parallel("Saving to results file at t=%f" % t)

        local_project(alpha_, Y2, alpha_r) # project alpha values to nodes
        file_results.write(T_, t)
        file_results.write(alpha_r, t)

    # Write to csv file
    front_pos_str = ", ".join(["{}".format(x_pos) for x_pos in front_loc])
    alpha_max_str = ", ".join(["{}".format(a_max) for a_max in alpha_max])
    print_file.write("{},{},{}\n".format(t, front_pos_str, alpha_max_str))
    print_file.flush()


    step += 1
