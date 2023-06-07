# Michael Zakoworotny, Gavin DeBrun
# Decoupled algorithm for printing of multimode viscoelastic fluid 
# Using a 3-step algorithm at each time step:
# 1) Compute new velocity, pressure and velocity gradient projection 
# by solving stokes system, using a first order semi-implicit prediction
# of the elastic stresses
# 2) Compute new elastic stresses using a BDF2 semi-implicit scheme, using
# a predictor of the stresses in the implicit terms to make this method 
# explicit. Using ALE form of the material derivative, and SUPG on free surface
# 3) Compute alpha and temperature using newly computed velocity, with ALE form
#  - Inlet problem solved on 1D mesh for faster convergence and better stability
#  - Computing mesh displacement by solving streamline equation on 1D mesh,
#    rather than manually using Gauss quadrature
# Multi-mode Giesekus fluid
# Prescribed print velocity at lower boundary once material has begun to cure
# Using a different mesh which is coarser inside the nozzle region
# Solving each stress equation independently
# Including thermal expansion and cure shrinkage

from dolfin import *
import os
import sys
import time
from math import ceil, floor, pi, tanh
import numpy as np
# import gmsh
import meshio
from scipy.interpolate import interp1d
from scipy.signal import square
from mpi4py import MPI
import csv
from ufl import tanh as tanh_ufl
from ufl import min_value, max_value

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

QUAD_DEG = 4#8
parameters["form_compiler"]["quadrature_degree"] = QUAD_DEG
dx = dx(degree=QUAD_DEG, scheme="default")

######################## PARAMETERS ########################
# Geometry
R = 0.00154/2
L_noz = R*2 # length of nozzle region
L_swell = 0.003 # length outside nozzle
# Mesh parameter
L_front_refine = 1e-3 # distance at which the mesh is refined for accurate solution of thermochemical equations
                         # make sure L_front_refine is less than the expected length of uncured filament, Lb

rho = 980   # density
# Rheological parameters
m = 3 # number of modes
eta = [73.4, 60.1, 15.1]
lam = [0.1, 0.91, 4.54]
alpha = [0.491, 0.193, 0.063]
eta_s = 0.0528

g_ = 9.81

# Thermo-chemical parameters
kr =  0.15
Cp = 1600.0
R_ = 8.314
# Nature values
Hr =  350000
A_ = 8.55e15
Er = 110750.0
n_ = 1.72
m_ = 0.77
Ca = 14.48
alpha_c = 0.41

# Viscosity evolution parameters (visc. increase w/ alpha)
eta1 = eta
eta2 = [e*1000 for e in eta] # scale viscosity by 3 orders of magnitude by alpha gel
a1 = 0.1
a2 = 0.4 #0.5
visc_delay = 0.1 # delay after which the viscosity increase will start being applied to system

# Printing parameters
v_gel_max = 1.0e-3
Q_flow = v_gel_max * pi*R**2
v_gel = lambda t : v_gel_max # instantaneous jump of flow rate

v_print_max = 0.8e-3
v_print = lambda t : v_print_max # instantaneous startup of print speed

# Thermo-chemical parameters
T0 = 20+273.15
alpha0 = 0.1
T_max = T0 + Hr/Cp*(1-alpha0)
time_trig = 7 # duration of trigger

# Heat convection
T_amb = 40+273.15
h_conv = 50 # W/m^2/K

therm_exp = 0.6e-3 # volumetric thermal expansion

# Time stepping parameters
tend = 60 #18 #12 # end of simulation
dt = 1e-3 #5e-4#0.01#0.01
num_steps = int(tend/dt)

# Output frequency
out_freq = 0.1
out_step = round(out_freq/dt)

######################## POISEUILLE FLOW 1D #####################

class Poiseuille_Problem_1D():

    def __init__(self, R, nel_r, eta, lam, alpha, eta_s, mesh=None):
        self.R = R
        self.nel_r = nel_r
        self.eta = eta
        self.lam = lam
        self.alpha = alpha
        self.m = len(eta)
        self.eta_s = eta_s
        # self.v_av = v_av

        self.setup_problem(mesh) # assemble mesh, variational form, solver

    def mesh_1D_surface(self):
        # Mesh geometry
        mesh_size = 0.1
        mesh_name = "nozzle_die_mesh_1D"
        gmsh.initialize()
        gmsh.model.add(mesh_name)
        p1 = gmsh.model.geo.addPoint(0.,0.,0,mesh_size)
        p2 = gmsh.model.geo.addPoint(0.,self.R,0,mesh_size)
        c1 = gmsh.model.geo.addLine(p1,p2) # bottom lines
        gmsh.model.geo.synchronize()
        # Meshing setup
        gmsh.model.geo.mesh.setTransfiniteCurve(c1, self.nel_r+1, coef=1)
        # Mesh and write to file
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(1)
        gmsh.write(mesh_name+".msh")
        # Import gmsh mesh - first write msh to xdmf, then import to fenics
        if comm_rank==0:
            msh = meshio.read(mesh_name+".msh")
            msh.prune_z_0()
            meshio.write(mesh_name+".xdmf", meshio.Mesh(points=msh.points, cells={"line":msh.get_cells_type("line")})) # add cell data if physical groups specified
        mesh = Mesh()
        with XDMFFile(mesh_name+".xdmf") as infile:
            infile.read(mesh)
        if comm_rank==0:
            os.remove(mesh_name+".msh")
            os.remove(mesh_name+".xdmf")
            os.remove(mesh_name+".h5")

        return mesh

    def setup_problem(self, mesh=None):
        # MESH
        if not mesh: # Only generate mesh if not supplied
            mesh = self.mesh_1D_surface()
        # Define boundaries
        bot = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=0.0, tol=1e-7)
        top = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=self.R, tol=1e-7)
        # Mark boundaries
        facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        facets.set_all(0)
        bot.mark(facets, 1)
        top.mark(facets, 2)
        ds = Measure('ds', domain=mesh, subdomain_data=facets) # partition boundary

        # Function space
        fe_v = FiniteElement("CG", mesh.ufl_cell(), 2)
        fe_p = FiniteElement("CG", mesh.ufl_cell(), 1)
        fe_s = TensorElement("CG", mesh.ufl_cell(), 1)
        fe_c = FiniteElement("Real", mesh.ufl_cell(), 0)
        elems = [fe_v, fe_c]
        for i in range(self.m):
            elems.append(fe_s)
        mixed_elem = MixedElement(elems)
        V = FunctionSpace(mesh, mixed_elem)
        # Functions
        te = TestFunctions(V)
        [w, mu1] = te[0:2]
        tau_list = te[2:]
        du_mixed = TrialFunction(V)
        u_mixed = Function(V)
        u_mixed_form = split(u_mixed) # functions for constucting forms
        [v, lag1] = u_mixed_form[0:2]
        s_list = u_mixed_form[2:]
        u_mixed_post = u_mixed.split() # functions for post-processing
        [v_, lag1_] = u_mixed_post[0:2]
        s_list_ = u_mixed_post[2:]
        v_.rename("Velocity","v")
        lag1_.rename("lagrange_multiplier1","lag1")
        for i in range(self.m):
            s_list_[i].rename("sigma{}".format(i+1),"s{}".format(i+1))
        u_n_mixed = Function(V) # solution at previous step
        u_n_mixed_form = split(u_n_mixed)
        [v_n, lag1_n] = u_n_mixed_form[0:2]
        s_n_list = u_n_mixed_form[2:]

        # Boundary conditions
        v_wall = Constant(0.)
        bc_wall = DirichletBC(V.sub(0), v_wall, facets, 2) # no slip wall
        bcs = [bc_wall]
        s_sym = Constant([[0.,0.],[0.,0.]])
        for i in range(self.m): # traction free axis
            bcs.append(DirichletBC(V.sub(2+i), s_sym, facets, 1))

        # Variational form
        x = MeshCoordinates(mesh)
        n = FacetNormal(mesh)
        v_flow = Constant("0.")
        gradv = as_tensor([[0, 0],[v.dx(1), 0]])
        gradv_n = as_tensor([[0, 0],[v_n.dx(1), 0]])
        # F = w*(eta_s*v.dx(1) + sum(s_list)[0,1] - 1/2*dpdx_*x[1])*dx
        F = w.dx(1)*x[1]*(eta_s*v.dx(1) + sum(s_list)[0,1])*dx #+ w*x[1]*dpdx_*dx # <- equivalent to the above
        for i in range(m):
            # F += inner(tau_list[i], (lam[i]*(-dot(gradv.T, s_list[i]) - dot(s_list[i],gradv)) + s_list[i] + alpha[i]*lam[i]/eta[i]*dot(s_list[i],s_list[i]) - eta[i]*(gradv + gradv.T)))*dx
            F += inner(tau_list[i], lam[i]*1/dt*(s_list[i]-s_n_list[i]) + 1/2*((lam[i]*(-dot(gradv.T,s_list[i]) - dot(s_list[i],gradv)) + s_list[i] + alpha[i]*lam[i]/eta[i]*dot(s_list[i],s_list[i]) - eta[i]*(gradv + gradv.T)) + \
                        (lam[i]*(-dot(gradv_n.T,s_n_list[i]) - dot(s_n_list[i],gradv_n)) + s_n_list[i] + alpha[i]*lam[i]/eta[i]*dot(s_n_list[i],s_n_list[i]) - eta[i]*(gradv_n + gradv_n.T))))*dx
        F += x[1]*w*lag1*dx + x[1]*mu1*v*dx - mu1*x[1]*v_flow*dx # lagrange multiplier on inlet flow rate

        tang = derivative(F, u_mixed, du_mixed)

        # Solver
        problem = NonlinearVariationalProblem(F, u_mixed, bcs=bcs, J=tang, form_compiler_parameters=ffc_options)
        solver = NonlinearVariationalSolver(problem)
        prms = solver.parameters
        prms["nonlinear_solver"] = "newton"
        prms["newton_solver"]["linear_solver"] = "default"
        prms["newton_solver"]["report"] = True
        prms["newton_solver"]["absolute_tolerance"] = 1e-14
        prms["newton_solver"]["relative_tolerance"] = 1e-14
        prms["newton_solver"]["error_on_nonconvergence"] = False
        prms["newton_solver"]["convergence_criterion"] = "residual"
        prms["newton_solver"]["maximum_iterations"] = 10

        file_results = XDMFFile("poiseuille_inlet_1D_transient.xdmf")
        file_results.parameters["flush_output"] = True
        file_results.parameters["functions_share_mesh"] = True

        S = TensorFunctionSpace(mesh, "CG", 1)
        cauchy_ = Function(S, name="cauchy_stress")

        self.solver = solver
        self.file_results = file_results
        self.v_flow = v_flow
        self.v_ = v_
        self.s_list_ = s_list_
        self.cauchy_ = cauchy_ # access function space by function_space()
        self.u_mixed = u_mixed
        self.u_n_mixed = u_n_mixed

    def solve_poiseuille(self, t, v_av):
        # Solve Poiseuille problem at a given time with average flow rate v_av
        print_parallel("Solving inlet poiseuille problem at time {}".format(t))

        self.v_flow.assign(v_av)
        self.u_n_mixed.assign(self.u_mixed)
        self.solver.solve()

        return self.v_, self.s_list_

    def write_file(self, t):
        self.cauchy_.assign(project(self.eta_s*(as_tensor([[0., self.v_.dx(1)],[self.v_.dx(1), 0.]])) + sum(self.s_list_), self.cauchy_.function_space()))

        self.file_results.write(self.v_, t)
        for i in range(self.m):
            self.file_results.write(self.s_list_[i], t)
        self.file_results.write(self.cauchy_, t)

    def undo_step(self):
        # function that removes the previously computed load step by reverting the function
        # to the values it had previously
        # NOT CORRECT FOR TRANSIENT (would need to reset u_n_mixed as well)
        self.u_mixed.assign(self.u_n_mixed)
        # Also seemingly need to redefine split functions ...
        u_mixed_post = self.u_mixed.split() # functions for post-processing
        [self.v_, self.lag1_] = u_mixed_post[0:2]
        self.s_list_ = u_mixed_post[2:]
        self.v_.rename("Velocity","v")
        self.lag1_.rename("lagrange_multiplier1","lag1")
        for i in range(self.m):
            self.s_list_[i].rename("sigma{}".format(i+1),"s{}".format(i+1))

######################## MESH GENERATION ########################
# Meshes pre-generated in separate script

######################## SETUP DIE SWELL PROBLEM ################

# Initialize object for solving for  inlet velocity function
nel_r_inlet = 100
mesh_inlet = Mesh()
with XDMFFile("meshes/meshInlet_R0p77.xdmf") as infile:
    infile.read(mesh_inlet)
inlet_problem = Poiseuille_Problem_1D(R, nel_r_inlet, eta, lam, alpha, eta_s, mesh=mesh_inlet)

# # Generate meshes
# hR = R/8 #R/64#R/25#R/12
# hRrefine1 = R/25 #R/64#R/25#R/25
# hRrefine2 = R/40 #R/25 #R/25 #R/36
# # mesh = refined_mesh_improved(hR, hRrefine1, hRrefine2)
# mesh = refined_mesh_unstructured(L_front_refine, hR, hRrefine1, hRrefine2)
# dx = Measure('dx', domain=mesh)
# # mesh_1D = refined_mesh_1D_surface(hRrefine2, hRrefine1)
# mesh_1D = mesh_1D_extract(mesh)
# dx_1D = Measure('dx', domain=mesh_1D)

# IMPORT PRE-GENERATED MESH
mesh_string = "V" + str(floor(v_print_max*1e3)) + "p" + str(round(((v_print_max*1e3)%1)*10)) + "_h" + str(round(h_conv)) + "_Tamb" + str(round(T_amb - 273.15))
mesh = Mesh()
with XDMFFile("meshes/mesh2D_" + mesh_string + ".xdmf") as infile:
    infile.read(mesh)
dx = Measure('dx', domain=mesh)
mesh_1D = Mesh()
with XDMFFile("meshes/meshSurf_" + mesh_string + ".xdmf") as infile:
    infile.read(mesh_1D)
dx_1D = Measure('dx', domain=mesh_1D)


# Define boundaries
left = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=0.0, tol=1e-7)
bot = CompiledSubDomain("on_boundary && near(x[1], side, tol)", side=0.0, tol=1e-7)
right = CompiledSubDomain("on_boundary && near(x[0], side, tol)", side=L_noz+L_swell, tol=1e-7)
top_swell = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]>=(dist-tol)", side=R, dist=L_noz, tol=1e-7)
top_noz = CompiledSubDomain("on_boundary && near(x[1], side, tol) && x[0]<=(dist+tol)", side=R, dist=L_noz, tol=1e-7)
# Mark boundaries
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
facets.set_all(0)
left.mark(facets, 1)
bot.mark(facets, 2)
right.mark(facets, 3)
top_swell.mark(facets, 4)
top_noz.mark(facets, 5)
ds = Measure('ds', domain=mesh, subdomain_data=facets) # partition boundary

# Define boundaries for 1D mesh
facets_1D = MeshFunction("size_t", mesh_1D, mesh_1D.topology().dim()-1, 0)
facets_1D.set_all(0)
top_noz.mark(facets_1D, 1)


# Function spaces for stokes problem
fe_v = VectorElement("CG", mesh.ufl_cell(), 2)
fe_p = FiniteElement("CG", mesh.ufl_cell(), 1)
fe_g = TensorElement("CG", mesh.ufl_cell(), 1)
fe_gt = FiniteElement("CG", mesh.ufl_cell(), 1)
fe_c = FiniteElement("Real", mesh.ufl_cell(), 0)
elems_stokes = [fe_v, fe_p, fe_g, fe_gt]
mixed_elem_stokes = MixedElement(elems_stokes)
V_stokes = FunctionSpace(mesh, mixed_elem_stokes)
# Functions
[w, q, E, Et] = TestFunctions(V_stokes)
[v, p, G, Gt] = TrialFunctions(V_stokes)
u_mixed = Function(V_stokes)
[v_, p_, G_, Gt_] = u_mixed.split() # output functions
v_.rename("Velocity","v")
p_.rename("Pressure","p")
G_.rename("velgrad_proj","G")
Gt_.rename("velgradt_proj","Gt")
u_n_mixed = Function(V_stokes)
[v_n, p_n, G_n, Gt_n] = u_n_mixed.split() # values at previous step

# Function spaces for constitutive equations
fe_s = TensorElement("CG", mesh.ufl_cell(), 1)
fe_st = FiniteElement("CG", mesh.ufl_cell(), 1)
V_const = [] # now each V_const is a separate function space
for i in range(m):
    mixed_elem_const = MixedElement([fe_s, fe_st])
    V_const.append(FunctionSpace(mesh, mixed_elem_const))
# Functions
tau_list = [TestFunctions(v)[0] for v in V_const]
taut_list = [TestFunctions(v)[1] for v in V_const]
s_list = [TrialFunctions(v)[0] for v in V_const]
st_list = [TrialFunctions(v)[1] for v in V_const]
s_mixed = [Function(v) for v in V_const]
s_list_ = [s.split()[0] for s in s_mixed] # can be output
st_list_ = [s.split()[1] for s in s_mixed]
for i in range(m):
    s_list_[i].rename("sigma{}".format(i+1),"s{}".format(i+1))
    st_list_[i].rename("sigmath{}".format(i+1),"sth{}".format(i+1))
s_n_mixed = [Function(v) for v in V_const]
s_n_list = [split(s)[0] for s in s_n_mixed] # stresses at previous time step
st_n_list = [split(s)[1] for s in s_n_mixed]
s_nn_mixed = [Function(v) for v in V_const] # stresses at second previous time step
s_nn_list  = [split(s)[0] for s in s_nn_mixed]
st_nn_list = [split(s)[1] for s in s_nn_mixed]

# Function spaces for thermo-chemical problem
V_therm = FunctionSpace(mesh, "CG", 1)
gam = TestFunction(V_therm)
T = TrialFunction(V_therm)
T_ = Function(V_therm, name="temp")
T_n = Function(V_therm)
T_nn = Function(V_therm)
V_alpha = FunctionSpace(mesh, "CG", 1)
beta = TestFunction(V_alpha)
al = TrialFunction(V_alpha)
al_ = Function(V_alpha, name="alpha")
al_n = Function(V_alpha)
al_nn = Function(V_alpha) # need alpha from two time steps ago for cure shrinkage function

# Functions for evaluating inlet boundary conditions
v_inlet_interp = Function(V_stokes.sub(0).sub(0).collapse(), name="v_inlet_interp")
s_inlet_interp = []
for i in range(m):
    s_inlet_interp.append(Function(V_const[i].sub(0).collapse(), name="s_inlet_interp"+str(i+1)))

# Evolution of viscosity with degree of cure - increase linearly on log scale
# Degree of cure as separate function space (DG0)
W3 = FunctionSpace(mesh, "DG", 0)
eta_fun = []
for i in range(m):
    eta_fun.append(Function(W3, name="eta{}".format(i+1))) # min_value(eta1[i]*(eta2[i]/eta1[i])**((al-a1)/(a2-a1)), eta2[i])
eta_devss = sum(eta_fun) # Stabilization parameter function

# Additional functions for post-processing
V_tens = TensorFunctionSpace(mesh, "CG", 1)
cauchy_ = Function(V_tens, name="cauchy_stress")
grad_v_ = Function(V_tens, name="grad(v)")

V_scal = FunctionSpace(mesh, "CG", 1)
cauchyt_ = Function(V_scal, name="cauchy_tt_stress")
div_v_plot = Function(V_scal, name="div_v") # projection of div_v for plotting

# Functions for displacement (moved here because need om_ALE for ALE derivative)
U2 = VectorFunctionSpace(mesh, "CG", 1)
xi2 = TestFunction(U2)
om = TrialFunction(U2)
om_ = Function(U2, name="displacement")
om_n = Function(U2) # displacement from previous time step
om_ALE = Function(U2) # the change in displacement that I move the mesh by


# Dirichlet boundary conditions for stokes problem
v_outlet = Expression(("v_print","0."),degree=2, v_print=0.)
bc_wall = DirichletBC(V_stokes.sub(0), Constant(("0.","0.")), facets, 5)
bc_sym = DirichletBC(V_stokes.sub(0).sub(1), Constant("0."), facets, 2)
bc_inlet_x = DirichletBC(V_stokes.sub(0).sub(0), v_inlet_interp, facets, 1)
bc_inlet_r = DirichletBC(V_stokes.sub(0).sub(1), Constant("0."), facets, 1)
bc_outlet = DirichletBC(V_stokes.sub(0), v_outlet, facets, 3)
bcs_stokes_uncured = [bc_wall, bc_sym, bc_inlet_x, bc_inlet_r] # no outlet BC before cured
bcs_stokes_cured = [bc_wall, bc_sym, bc_inlet_x, bc_inlet_r, bc_outlet]

# Dirichlet boundary conditions for constitutive problem
bcs_const = []
for i in range(m):
    bc_s_inlet = DirichletBC(V_const[i].sub(0), s_inlet_interp[i], facets, 1)
    bc_st_inlet = DirichletBC(V_const[i].sub(1), Constant("0."), facets, 1)
    bcs_const.append([bc_s_inlet, bc_st_inlet])

# Dirichlet boundary conditions for thermo-chemical problem
bc_temp_inlet = DirichletBC(V_therm, Constant(T0), facets, 1)
bc_temp_noz = DirichletBC(V_therm, Constant(T0), facets, 5)
bc_temp_trigger = DirichletBC(V_therm, Constant(T_max), facets, 3)
bcs_temp_trigger = [bc_temp_inlet, bc_temp_noz, bc_temp_trigger]
bcs_temp = [bc_temp_inlet, bc_temp_noz]

bc_alpha_inlet = DirichletBC(V_alpha, Constant(alpha0), facets, 1)
bc_alpha_noz = DirichletBC(V_alpha, Constant(alpha0), facets, 5)
bcs_alpha = [bc_alpha_inlet, bc_alpha_noz]

# Initial conditions for temperature and alpha
T_.assign(project(Constant(T0), V_therm))
T_n.assign(project(Constant(T0), V_therm))
T_nn.assign(project(Constant(T0), V_therm))
al_.assign(project(Constant(alpha0), V_alpha))
al_n.assign(project(Constant(alpha0), V_alpha))
al_nn.assign(project(Constant(alpha0), V_alpha))

# Thermal expansion and cure shrinkage
F_therm = 1 + therm_exp*(T_n - T0) # T_n equal T_ when solving stokes flow problem
F_therm_deriv = therm_exp          # d(F_therm)/dT
F_cure = Function(V_alpha)         # use same function space as alpha to do vector operations
F_cure_deriv = Function(V_alpha)

# Polynomial fit to experimental data
def update_cure(F_cure, al):
    # F_cure.vector()[:] = np.logical_and(al.vector()[:]>=0.2817, al.vector()[:]<0.8)*(1 - 16.3794*al.vector()[:]**5 + 50.5869*al.vector()[:]**4 - 61.9261*al.vector()[:]**3 + 37.5613*al.vector()[:]**2 - 11.338*al.vector()[:] + 1.3080) \
    #                         + (al.vector()[:]>=0.8)*(1 - 16.3794*0.8**5 + 50.5869*0.8**4 - 61.9261*0.8**3 + 37.5613*0.8**2 - 11.338*0.8 + 1.3080) \
    #                         + (al.vector()[:]<0.2817)*1
    F_cure.vector()[:] = -(1 - 0.923861248)/2 * np.tanh(10*(al.vector()[:] - (0.8 + 0.2816399740898)/2)) + (1 + 0.923861248)/2

def update_cure_deriv(F_cure_deriv, al):
    # F_cure_deriv.vector()[:] = np.logical_and(al.vector()[:]>=0.2817, al.vector()[:]<0.8)*(-5*16.3794*al.vector()[:]**4 + 4*50.5869*al.vector()[:]**3 - 3*61.9261*al.vector()[:]**2 + 2*37.5613*al.vector()[:] - 11.338) \
    #                             + np.logical_or(al.vector()[:]<0.2817, al.vector()[:]>=0.8)*0
    F_cure_deriv.vector()[:] = -(1 - 0.923861248)/2 * 10 * 1/np.cosh(10*(al.vector()[:] - (0.8 + 0.2816399740898)/2))**2

# Quantities for variational form
h = CellDiameter(mesh)
x = MeshCoordinates(mesh)
n = FacetNormal(mesh)
g_vec = Expression(("g","0."), degree=1, g=0)

# Variational form for stokes problem
# 1st order prediction of stresses for stokes problem
s_predict_stokes_list = []
st_predict_stokes_list = []
for i in range(m):
    s_predict_stokes = s_n_list[i] - dt*(dot(v - om_ALE/dt, nabla_grad(s_n_list[i])) - dot(G.T, s_n_list[i]) - dot(s_n_list[i], G) + (tr(G) + Gt)*s_n_list[i]) \
                                   - dt/lam[i]*(s_n_list[i] + alpha[i]*lam[i]/eta_fun[i]*dot(s_n_list[i], s_n_list[i]) - eta_fun[i]*(G+G.T - 2/3*(tr(G) + Gt)*Identity(2)))
    s_predict_stokes_list.append(s_predict_stokes)
    st_predict_stokes = st_n_list[i] - dt*(dot(v - om_ALE/dt, nabla_grad(st_n_list[i])) - 2*Gt*st_n_list[i] + (tr(G) + Gt)*st_n_list[i]) \
                                     - dt/lam[i]*(st_n_list[i] + alpha[i]*lam[i]/eta_fun[i]*st_n_list[i]**2 - eta_fun[i]*(2*Gt - 2/3*(tr(G) + Gt)))
    st_predict_stokes_list.append(st_predict_stokes)
# Rate of deformation and total stress
gammadot = nabla_grad(v) + nabla_grad(v).T - 2/3*(div(v) + v[1]/x[1])*Identity(2)
gammadot_tt = 2*v[1]/x[1] - 2/3*(div(v) + v[1]/x[1])
T_str = -p*Identity(2) + eta_s*gammadot + sum(s_predict_stokes_list)
T_tt_str = -p + eta_s*gammadot_tt + sum(st_predict_stokes_list)
# Momentum, contintuity and gradient projection equations
F_s = x[1]*inner(nabla_grad(w), T_str)*dx + w[1]*T_tt_str*dx - x[1]*rho*dot(w, g_vec)*dx
F_s += x[1]*inner(nabla_grad(w), eta_devss*(gammadot - (G + G.T - 2/3*(tr(G) + Gt)*Identity(2))))*dx + w[1]*eta_devss*(gammadot_tt - (2*Gt - 2/3*(tr(G) + Gt)))*dx # DEVSS-G stabilization
F_s += x[1]*q*(div(v) + v[1]/x[1])*dx - x[1]*q*(1/F_therm*F_therm_deriv*((T_n - T_nn)/dt + dot(v_ - om_ALE/dt, nabla_grad(T_n))) + 1/F_cure*F_cure_deriv*((al_n - al_nn)/dt + dot(v_ - om_ALE/dt, nabla_grad(al_n))))*dx # continuity
F_s += x[1]*inner(E, G-nabla_grad(v))*dx # gradient projection
F_s += x[1]*inner(Et, Gt-v[1]/x[1])*dx # gradient projection theta

# Variational form for constitutive equations
stablz = h/2/sqrt(dot(v_,v_)) # SUPG stabilization parameter
# First order semi-implicit for first time step
F_c_1 = []
for i in range(m):
    F_c_1_i = x[1]*inner(tau_list[i] + stablz*dot(v_ - om_ALE/dt, nabla_grad(tau_list[i])), lam[i]*(1/dt*(s_list[i] - s_n_list[i]) + dot(v_ - om_ALE/dt, nabla_grad(s_list[i])) \
                                - dot(G_.T, s_n_list[i]) - dot(s_n_list[i], G_) + (tr(G_) + Gt_)*s_n_list[i]) + s_n_list[i] + alpha[i]*lam[i]/eta_fun[i]*dot(s_n_list[i], s_n_list[i]) \
                                - eta_fun[i]*(G_ + G_.T - 2/3*(tr(G_) + Gt_)*Identity(2)))*dx
    F_c_1_i += x[1]*(taut_list[i] + stablz*dot(v_ - om_ALE/dt, nabla_grad(taut_list[i]))) * (lam[i]*(1/dt*(st_list[i] - st_n_list[i]) + dot(v_ - om_ALE/dt, nabla_grad(st_list[i])) \
                                - 2*Gt_*st_n_list[i] + (tr(G_) + Gt_)*st_n_list[i]) + st_n_list[i] + alpha[i]*lam[i]/eta_fun[i]*st_n_list[i]**2 - eta_fun[i]*(2*Gt_ - 2/3*(tr(G_) + Gt_)))*dx
    F_c_1.append(F_c_1_i)
# BDF2 semi-implicit for subsequent time steps
F_c_2 = []
for i in range(m):
    s_predict = 2*s_n_list[i] - s_nn_list[i]
    st_predict = 2*st_n_list[i] - st_nn_list[i]

    F_c_2_i = x[1]*inner(tau_list[i] + stablz*dot(v_ - om_ALE/dt, nabla_grad(tau_list[i])), lam[i]*(1/dt*(3/2*s_list[i] - 2*s_n_list[i] + 1/2*s_nn_list[i]) + dot(v_ - om_ALE/dt, nabla_grad(s_list[i])) \
                                - dot(G_.T, s_predict) - dot(s_predict, G_) + (tr(G_) + Gt_)*s_predict) + s_predict + alpha[i]*lam[i]/eta_fun[i]*dot(s_predict, s_predict) \
                                - eta_fun[i]*(G_ + G_.T - 2/3*(tr(G_) + Gt_)*Identity(2)))*dx
    F_c_2_i += x[1]*(taut_list[i] + stablz*dot(v_ - om_ALE/dt, nabla_grad(taut_list[i]))) * (lam[i]*(1/dt*(3/2*st_list[i] - 2*st_n_list[i] + 1/2*st_nn_list[i]) + dot(v_ - om_ALE/dt, nabla_grad(st_list[i])) \
                                - 2*Gt_*st_predict + (tr(G_) + Gt_)*st_predict) + st_predict + alpha[i]*lam[i]/eta_fun[i]*st_predict**2 - eta_fun[i]*(2*Gt_ - 2/3*(tr(G_) + Gt_)))*dx
    F_c_2.append(F_c_2_i)

# Variational form for degree of cure problem
tau_a = stablz
# tau_a = 1/sqrt((2*sqrt(dot(v_,v_))/h)**2 + (2/dt)**2)
# g = (1-al_n)**n_
g = (1-al_n)**n_ * al_n**m_ / (1+exp(Ca*(al_n-alpha_c)))
# F_a = x[1]*(beta + tau_a*dot(v_ - om_ALE/dt, grad(beta)))*((al - al_n)/dt + dot(v_ - om_ALE/dt, grad(al)) - A_*exp(-Er/R_/T_n)*g)*dx
# F_a = x[1]*(beta + tau_a*dot(v_ - om_ALE/dt, grad(beta)))*((al - al_n)/dt + dot(v_ - om_ALE/dt, grad(al_n)) - A_*exp(-Er/R_/T_)*g)*dx
# ONLY APPLYING SUPG TEST FN TO CONVECTIVE TERM
F_a = x[1]*beta*((al - al_n)/dt - A_*exp(-Er/R_/T_n)*g)*dx + x[1]*(beta + tau_a*dot(v_ - om_ALE/dt, grad(beta)))*(dot(v_ - om_ALE/dt, grad(al)))*dx

# Variational form for thermal problem
Pe = sqrt(dot(v_,v_))*h/2/(kr/Cp/rho) # Peclet number for thermal problem
tau_T = stablz * (1/tanh_ufl(Pe) - 1/Pe)
# tau_T = 1/sqrt((2*sqrt(dot(v_,v_))/h)**2 + (2/dt)**2 + 9*(4*(kr/Cp/rho)/h**2)**2)
F_T = x[1]*(kr*dot(grad(gam),grad(T)) + (gam + tau_T*dot(v_ - om_ALE/dt, grad(gam)))*(rho*Cp*((T - T_n)/dt + dot(v_ - om_ALE/dt, grad(T))) \
                                - rho*Hr*((al_ - al_n)/dt + dot(v_ - om_ALE/dt, grad(al_)))))*dx + x[1]*gam*h_conv*(T - T_amb)*ds(4)

# Solver for stokes when not cured
problem_stokes_uncured = LinearVariationalProblem(lhs(F_s), rhs(F_s), u_mixed, bcs_stokes_uncured, form_compiler_parameters=ffc_options)
solver_stokes_uncured = LinearVariationalSolver(problem_stokes_uncured)
# Solver for stokes when cured
problem_stokes_cured = LinearVariationalProblem(lhs(F_s), rhs(F_s), u_mixed, bcs_stokes_cured, form_compiler_parameters=ffc_options)
solver_stokes_cured = LinearVariationalSolver(problem_stokes_cured)

# Solver for constitutives at first time step
solver_const_1 = []
for i in range(m):
    problem_const_1 = LinearVariationalProblem(lhs(F_c_1[i]), rhs(F_c_1[i]), s_mixed[i], bcs_const[i], form_compiler_parameters=ffc_options)
    solver_const_1.append(LinearVariationalSolver(problem_const_1))
# Solver for constitutives at subsequent step
solver_const_2 = []
for i in range(m):
    problem_const_2 = LinearVariationalProblem(lhs(F_c_2[i]), rhs(F_c_2[i]), s_mixed[i], bcs_const[i], form_compiler_parameters=ffc_options)
    solver_const_2.append(LinearVariationalSolver(problem_const_2))

# Solver for degree of cure
problem_alpha = LinearVariationalProblem(lhs(F_a), rhs(F_a), al_, bcs_alpha, form_compiler_parameters=ffc_options)
solver_alpha = LinearVariationalSolver(problem_alpha)

# Solver for temperature with trigger
problem_temp_trigger = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs_temp_trigger, form_compiler_parameters=ffc_options)
solver_temp_trigger = LinearVariationalSolver(problem_temp_trigger)
# Solver for temperature without trigger
problem_temp = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs_temp, form_compiler_parameters=ffc_options)
solver_temp = LinearVariationalSolver(problem_temp)

############################## SETUP MESH DISPLACEMENT PROBLEM #################
# Store velocity DOF indices that are on free surface
dof_coords = V_stokes.tabulate_dof_coordinates() # coords of all DOF in space
vx_dofs_glob = np.array(V_stokes.sub(0).sub(0).dofmap().dofs()) # dof indices for vx <- these give the DOF among all processors
vr_dofs_glob = np.array(V_stokes.sub(0).sub(1).dofmap().dofs()) # dof indices for vr
all_dofs = V_stokes.dofmap().dofs()
vx_dofs = np.flatnonzero(np.isin(all_dofs, vx_dofs_glob))  # <- these give the DOF number for this processor
vr_dofs = np.flatnonzero(np.isin(all_dofs, vr_dofs_glob))
vx_surf_dofs = vx_dofs[np.where(np.logical_and(dof_coords[vx_dofs, 1]>=R, dof_coords[vx_dofs, 0]>=L_noz))[0]] # dof for vx on surface
vr_surf_dofs = vr_dofs[np.where(np.logical_and(dof_coords[vr_dofs, 1]>=R, dof_coords[vr_dofs, 0]>=L_noz))[0]] # dof for vr on surface

# Setup problem for solution of free surface displacement location on 1D mesh
U_1D = FunctionSpace(mesh_1D, "CG", 2)
xi_1D = TestFunction(U_1D)
u_1D = TrialFunction(U_1D)
u_1D_ = Function(U_1D, name="disp_1D")
u_1D_n = Function(U_1D)
vx_1D = Function(U_1D)
vr_1D = Function(U_1D)
# F_mesh_1D = xi_1D * (u_1D + dt*vx_1D*u_1D.dx(0) - dt*vr_1D - u_1D_n)*dx_1D # first-order time discretization of mesh update
F_mesh_1D = (xi_1D + CellDiameter(mesh_1D)/2/vx_1D*vx_1D*xi_1D.dx(0)) * (u_1D + dt*vx_1D*u_1D.dx(0) - dt*vr_1D - u_1D_n)*dx_1D # first-order time discretization of mesh update
bcs_mesh_1D = [DirichletBC(U_1D, Constant(0.), facets_1D, 1)]
problem_mesh_1D = LinearVariationalProblem(lhs(F_mesh_1D), rhs(F_mesh_1D), u_1D_, bcs_mesh_1D, form_compiler_parameters=ffc_options)
solver_mesh_1D = LinearVariationalSolver(problem_mesh_1D)

# DOF indices in parallel for 1D mesh
dof_coords_1D = U_1D.tabulate_dof_coordinates()
u_1D_dofs = np.array(U_1D.dofmap().dofs())

# Solve for linear variation of displacement in r direction
U1 = FunctionSpace(mesh, "CG", 1)
om_bound = Function(U1, name="disp_top") # solution of vertical displacement on top surface, applied to DOF on top surface
# U2 = VectorFunctionSpace(mesh, "CG", 1)
# xi2 = TestFunction(U2)
# om = TrialFunction(U2)
# om_ = Function(U2, name="displacement")
# om_n = Function(U2) # displacement from previous time step
# om_ALE = Function(U2) # the change in displacement that I move the mesh by
bcs_mesh_2D = [DirichletBC(U2, Constant((0.,0.)), facets, 2), DirichletBC(U2, Constant((0.,0.)), facets, 5), DirichletBC(U2.sub(1), om_bound, facets, 4), DirichletBC(U2.sub(0), Constant(0.), facets, 4)]
F_mesh_2D = xi2[1].dx(1)*om[1].dx(1)*dx - xi2[1]*om[1].dx(1)*n[1]*ds + xi2[0].dx(1)*om[0].dx(1)*dx
problem_mesh_2D = LinearVariationalProblem(lhs(F_mesh_2D), rhs(F_mesh_2D), om_, bcs_mesh_2D, form_compiler_parameters=ffc_options)
solver_mesh_2D = LinearVariationalSolver(problem_mesh_2D)

# Store DOF indices for om_bound that are on free surface
dof_coords_om = U1.tabulate_dof_coordinates() # coords of all DOF in space
om_dofs_glob = np.array(U1.dofmap().dofs()) # dof indices for om_bound <- these give the global DOF among all processors
all_dofs_om = U1.dofmap().dofs()
om_dofs = np.flatnonzero(np.isin(all_dofs_om, om_dofs_glob))  # <- these give the DOF number for this processor
om_surf_dofs = om_dofs[np.where(np.logical_and(dof_coords_om[om_dofs, 1]>=R, dof_coords_om[om_dofs, 0]>=L_noz))[0]] # dof for om_bound on surface

# Get dof id of swelling point, and corresponding comm
swell_id_sub = np.where(np.logical_and(np.isclose(mesh.coordinates()[:,0],L_noz+L_swell,atol=1e-10,rtol=1e-10), np.isclose(mesh.coordinates()[:,1],R,atol=1e-10,rtol=1e-10)))[0]
swell_comm_sub = np.array((comm_rank,))
try:
    swell_id_sub[0]
except IndexError:
    swell_id_sub = np.array((10000000,))
    swell_comm_sub = np.array((10000000,))
swell_id = np.zeros_like(swell_id_sub)
comm.Allreduce(swell_id_sub, swell_id, op=MPI.MIN)
swell_id = swell_id[0]
swell_comm = np.zeros_like(swell_comm_sub)
comm.Allreduce(swell_comm_sub, swell_comm, op=MPI.MIN)
swell_comm = swell_comm[0]

def get_surface_velocities(vel, velx_surf_dofs, velr_surf_dofs):
    # Return sorted arrays of velocity components on free surface, and corresponding x coordinates
    x_coord_x_gather = comm.gather(dof_coords[velx_surf_dofs,0], root=0)
    x_coord_r_gather = comm.gather(dof_coords[velr_surf_dofs,0], root=0)
    vx_gather = comm.gather(vel.vector()[velx_surf_dofs], root=0)
    vr_gather = comm.gather(vel.vector()[velr_surf_dofs], root=0)
    if comm_rank==0:
        x_coord_x_gather = np.concatenate(x_coord_x_gather)
        x_coord_x_sort_inds = x_coord_x_gather.argsort()
        x_coord_x_gather = x_coord_x_gather[x_coord_x_sort_inds]
        vx_gather = np.concatenate(vx_gather)[x_coord_x_sort_inds]
        x_coord_r_gather = np.concatenate(x_coord_r_gather)
        x_coord_r_sort_inds = x_coord_r_gather.argsort()
        x_coord_r_gather = x_coord_r_gather[x_coord_r_sort_inds]
        vr_gather = np.concatenate(vr_gather)[x_coord_r_sort_inds]
    return comm.bcast(x_coord_x_gather), comm.bcast(vx_gather), comm.bcast(vr_gather)

def get_die_swell(disp, swell_end_id, swell_end_comm):
    # Return displacement of upper-right node
    # Get coordinate of upper-right point in parallel, to ensure correct evaluation
    if comm_rank == swell_end_comm:
        vert_vals = np.reshape(disp.compute_vertex_values(),(-1,2),"F")
        disp_cur_sub = np.array((vert_vals[swell_end_id,1],))
        coord_cur_sub = np.array((mesh.coordinates()[swell_end_id,1],))
    else:
        disp_cur_sub = np.inf*np.ones((1,))
        coord_cur_sub = np.inf*np.ones((1,))
    disp_cur = np.zeros_like(disp_cur_sub)
    comm.Allreduce(disp_cur_sub, disp_cur, op=MPI.MIN)
    coord_cur = np.zeros_like(coord_cur_sub)
    comm.Allreduce(coord_cur_sub, coord_cur, op=MPI.MIN)
    disp_cur = disp_cur[0]
    coord_cur = coord_cur[0]
    
    return disp_cur, 1+disp_cur/R

########################## SOLVER ##########################################

# Setup results file
file_results = XDMFFile("results_transient_print.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(v_,0)
file_results.write(p_,0)
file_results.write(G_,0)
file_results.write(Gt_,0)
file_results.write(T_,0)
file_results.write(al_,0)

# Setup die swell information file
swell_file = open("swell_data_visco_transient.csv",'w')
swell_file.write("t,v_flow,r_coord,disp,swell,Q_in,Q_out,Q_surf\n")

outlet_cure = True # set to True when outlet has reached cure (switch to Vprint BC)

t = 0
step = 1
start_time = time.time()
while step < num_steps+1:
    t += dt
    print_parallel("t = %f, time elapsed = %f" % (t, time.time()-start_time))

    # Update inlet velocity
    v_gel_cur = v_gel(t)
    v_profile, s_profile_list = inlet_problem.solve_poiseuille(t, v_gel_cur)
    LagrangeInterpolator.interpolate(v_inlet_interp, v_profile)
    for j in range(m):
        LagrangeInterpolator.interpolate(s_inlet_interp[j], s_profile_list[j])

    v_print_cur = v_print(t)
    v_outlet.v_print = v_print_cur

    # UPDATE FLUID QUANTITIES USING THERMO-CHEMICAL RESULTS - project modal viscosities onto DG0 function
    if t >= visc_delay:
        for j in range(m):
            # eta_fun[j].assign(project(eta[j], W3))
            eta_fun[j].assign(project(min_value(eta1[j]*(eta2[j]/eta1[j])**((al_-a1)/(a2-a1)), eta2[j]), W3))
    else:
        for j in range(m):
            eta_fun[j].assign(project(eta[j], W3))

    if t >= visc_delay:
        g_vec.g = g_


    # Update cure shrinkage strains
    update_cure(F_cure, al_)
    update_cure_deriv(F_cure_deriv, al_)

    # print_parallel("\t Before stokes, time elapsed = %f" % (time.time()-start_time))
    # Solve fluid field for v_n+1, p_n+1, G_n+1
    if outlet_cure: # Vprint BC at outlet boundary
        solver_stokes_cured.solve()
    else: # free surface BC at outlet boundary
        solver_stokes_uncured.solve()
    # print_parallel("\t Before const, time elapsed = %f" % (time.time()-start_time))
    # Solve stress fields for s_n+1
    if step == 1:
        for j in range(m):
            solver_const_1[j].solve()
    else:
        for j in range(m):
            solver_const_2[j].solve()
    # print_parallel("\t Before alpha, time elapsed = %f" % (time.time()-start_time))
    # Solve alpha field for alpha_n+1
    solver_alpha.solve()
    al_.vector()[:] = np.minimum(al_.vector()[:],0.999)
    al_.vector()[:] = np.maximum(al_.vector()[:],alpha0-1e-4)
    # print_parallel("\t Before temp, time elapsed = %f" % (time.time()-start_time))
    # Solve temperature field for T_n+1
    if t<=time_trig: # Tmax at outlet boundary
        solver_temp_trigger.solve()
    else:
        solver_temp.solve()


    ## Correct mesh displacement using new solution, to get u_n+1 - Solve for displacement in domain based on free surface, store result in om_
    # Extract velocity components on free surface
    (x_coord_surf, vx_surf, vr_surf) = get_surface_velocities(v_, vx_surf_dofs, vr_surf_dofs)
    # Apply velocity components to functions on 1D mesh
    vx_1D.vector()[:] = vx_surf[np.argmin(abs(np.reshape(U_1D.tabulate_dof_coordinates()[:,0],(-1,1)) - np.reshape(x_coord_surf,(1,-1))),axis=1)]
    vr_1D.vector()[:] = vr_surf[np.argmin(abs(np.reshape(U_1D.tabulate_dof_coordinates()[:,0],(-1,1)) - np.reshape(x_coord_surf,(1,-1))),axis=1)]
    # Solve for new displacements on 1D domain (check if I need to do something about SUPG stabilization)
    solver_mesh_1D.solve()
    u_1D_n.assign(u_1D_) # update for next time step
    # Gather portions of 1D displacement from each core
    x_coord_1D_gather = comm.gather(dof_coords_1D[:,0], root=0)
    omega_surf_gather = comm.gather(u_1D_.vector()[:], root=0)
    omega_surf = []
    if comm_rank == 0:
        x_coord_1D_gather = np.concatenate(x_coord_1D_gather)
        x_coord_1D_sort_inds = x_coord_1D_gather.argsort()
        x_coord_1D_gather = x_coord_1D_gather[x_coord_1D_sort_inds]
        omega_surf = np.concatenate(omega_surf_gather)[x_coord_1D_sort_inds]
    x_coord_surf = comm.bcast(x_coord_1D_gather)
    omega_surf = comm.bcast(omega_surf) # broadcast surface displacement to all processes
    # Apply 1D displacement solution to 2D problem - prescribe values of surface solution to full function (assign displacements to closest points in function space)
    om_bound.vector()[om_surf_dofs] = omega_surf[np.argmin(abs(np.reshape(dof_coords_om[om_surf_dofs,0],(-1,1)) - np.reshape(x_coord_surf,(1,-1))),axis=1)]
    # Solve for displacement vector field for mesh motion, store in om_
    solver_mesh_2D.solve()
    # Move mesh by difference from last mesh movement
    # om_.vector()[:] = om_n.vector()[:] + 0.5*(om_.vector()[:] - om_n.vector()[:]) # Relaxation parameter for displacement solution (should not be needed in transient)
    om_ALE.assign(om_ - om_n)
    ALE.move(mesh, om_ALE)
    om_n.assign(om_)

    v_compute_inlet = assemble(2/R**2*x[1]*v_[0]*ds(1))
    print_parallel("\tAverage inlet velocity: {}".format(v_compute_inlet))
    Q_compute_inlet = assemble(2*pi*x[1]*v_[0]*ds(1))
    print_parallel("\tInlet flow rate: {}".format(Q_compute_inlet))
    Q_compute_outlet = assemble(2*pi*x[1]*dot(v_,n)*ds(3))
    print_parallel("\tOutlet flow rate: {}".format(Q_compute_outlet))
    Q_compute_surf = assemble(2*pi*dot(v_,n)*x[1]*ds(4))
    print_parallel("\tSurface flow rate: {}".format(Q_compute_surf))
    # v_normal = assemble(abs(dot(v_,n))*ds(4))
    # print_parallel("\tKinematic condition: {}".format(v_normal))

    disp_right, swell_right = get_die_swell(om_, swell_id, swell_comm) # evaluate displacement at upper right node
    swell_file.write("{},{},{},{},{},{},{},{}\n".format(t, v_gel_cur, disp_right+R, disp_right, swell_right, Q_compute_inlet, Q_compute_outlet, Q_compute_surf)) #t,v_flow,r_coord,disp,swell,Q_in,Q_out,Q_surf
    swell_file.flush()
    print_parallel("\tDisplacement: {}, swell ratio:{}".format(disp_right, swell_right))

    if step % out_step == 0:
        print_parallel("Saving to results file at t=%f" % t)

        inlet_problem.write_file(t)

        file_results.write(v_, t)
        file_results.write(p_, t)
        file_results.write(G_, t)
        file_results.write(Gt_, t)
        file_results.write(om_, t)
        file_results.write(T_, t)
        file_results.write(al_, t)

        cauchy_.assign(project(eta_s*(grad(v_) + grad(v_).T) + sum(s_list_), V_tens))
        cauchyt_.assign(project(2*eta_s*v_[1]/x[1] + sum(st_list_), V_scal))
        div_v_plot.assign(project(div(v_) + v_[1]/x[1], V_scal))

        file_results.write(cauchy_, t)
        file_results.write(cauchyt_, t)
        file_results.write(div_v_plot, t)

        for j in range(m):
            file_results.write(eta_fun[j], t)


    step += 1

    for j in range(m):
        s_nn_mixed[j].assign(s_n_mixed[j])
        s_n_mixed[j].assign(s_mixed[j])
    u_n_mixed.assign(u_mixed)
    al_nn.assign(al_n)
    al_n.assign(al_)
    T_nn.assign(T_n)
    T_n.assign(T_)
