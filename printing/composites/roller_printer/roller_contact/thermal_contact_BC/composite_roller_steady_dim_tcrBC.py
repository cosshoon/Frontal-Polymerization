# Michael Zakoworotny, Gavin DeBrun
# Thermo-chemical model for printing of continuous fiber composites,
# where impregnated composite tow is extruded through rollers
# Steady model, DIMENSIONAL model
# Using thermal contact resistance boundary condition

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
    if comm_rank == 0:
        print(mssg)
        sys.stdout.flush()


set_log_level(40)  # error level=40, warning level=30, info level=20
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
phi_1 = 0.2  # initial fiber volume fraction of impregnated tow (before compaction)
phi_2_target = 0.3

# Geometry
R_r = 5e-3  # 40e-3 # estimate roller diameter as 80 mm
L_up = 10e-3  # length of tow upstream of left roller contact point (ie. domain starts at x = -(L_up + l_r) )
L_down = 15e-3  # length of tow downstream of roller contact point
L = L_up + L_down  # total length of tow domain

# DCPD resin properties
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
kf = 6.83
Cpf = 1129.0

# Initial conditions of tow
T0 = 40 + 273.15  # initial temperature of tow
alpha0 = 0.1  # initial degree of cure of resin in impregnated tow

# Environmental conditions (convection)
h_conv = 50  # heat convection coefficient
Tamb = 30 + 273.15  # ambient temperature

# Thermal contact resistance (R_tc, ie. q = 1/R_tc*(T-Tamb))
# As R_tc -> 0, reduces to perfect thermal contact (T - Tamb -> 0)
R_tc = 1e-3  # K-m^2/W


# Process parameters (controlled by printing system)
V_r = 1e-3  # om_r * R_r # linear speed of surface of roller / tow

for i, V_r in enumerate([1e-3]):
    om_r = V_r / R_r  # 0.015 # rotational speed of roller
    T_r = (
        140 + 273.15
    )  # temperature of roller (assuming perfect contact between roller and tow)
    H_gap = 1e-3  # gap spacing between rollers

    H_tow = phi_2_target / phi_1 * H_gap

    # Derived quantities
    phi_2 = H_tow / H_gap * phi_1  # compacted fiber volume fraction
    dH = (H_tow - H_gap) / 2  # amount of compaction of tow (on each side of roller)
    a_r = acos(
        1 - dH / R_r
    )  # contact angle (in radians) between the roller and the tow
    # a_r = acos(1 - H_gap/(2*R_r)*(phi_2/phi_1 - 1))
    l_r = a_r * R_r  # linear contact length between roller and tow

    for j, R_tc in enumerate([1e-4]):
        # Steady solution parameters
        # loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # load ramping values (ramping roller temperature)
        load_dT = 5  # gap spacing in temperature between loads
        loads = (np.arange(T0, T_r + load_dT, load_dT) - T0) / (T_r - T0)
        iter_tol = 1e-10  # tolerance on steady solution
        max_iter = 2000  # 100 # maximum number of iterations per load step

        solve_noRoller = False  # whether or not to solve for steady state position when heating is removed

        # Homogenized properties - as functions of input phi
        k_par_fun = (
            lambda p: kr * (1 - p) + kf * p
        )  # thermal cond. parallel to fibers (x axis)
        k_perp_fun = lambda p: kr + 2 * p * kr / (
            (kf + kr) / (kf - kr)
            - p
            + (kf - kr) / (kf + kr) * (0.30584 * p**4 + 0.013363 * p**8)
        )  # thermal cond. perpendicular to fibers (y axis)
        rho_bar_fun = lambda p: rhor * (1 - p) + rhof * p  # homogenized density
        Cp_bar_fun = lambda p: (Cpr * rhor * (1 - p) + Cpf * rhof * p) / rho_bar_fun(
            p
        )  # homogenized specific heat
        rho_Cp_bar_fun = (
            lambda p: Cpr * rhor * (1 - p) + Cpf * rhof * p
        )  # product of homogenized density and specific heat

        # Characteristic scales
        phi0 = phi_2  # use phi2 as the reference fiber volume fraction
        T_max = T0 + rhor * Hr / rho_Cp_bar_fun(phi0) * (1 - phi0) * (1 - alpha0)
        v_c = (
            A_
            * R_
            * k_par_fun(phi0)
            * T_max**2
            / rhor
            / Hr
            / Er
            / (1 - phi0)
            / (1 - alpha0)
            * np.exp(-Er / R_ / T_max)
        ) ** (1 / 2)
        L_c = k_par_fun(phi0) / rho_Cp_bar_fun(phi0) / v_c
        t_c = L_c / v_c
        T_c = T_max

        # Nondimensional properties
        L_up_star = L_up / L_c
        L_down_star = L_down / L_c
        l_r_star = l_r / L_c
        H_gap_star = H_gap / L_c
        H_tow_star = H_tow / L_c
        R_r_star = R_r / L_c
        theta_r = (T_r - T0) / (T_c - T0)
        theta_amb = (Tamb - T0) / (T_c - T0)
        v_star = V_r / v_c

        write_str = "V{}_H{}_R{}".format(
            str(round(V_r * 1000, 2)).replace(".", "p"),
            str(round(H_gap * 1000, 2)).replace(".", "p"),
            str(round(R_r * 1000, 2)).replace(".", "p"),
        )

        ################################### SETUP PROBLEM ######################################

        h_elem = 2e-5
        # nel_x = ceil(L/h_elem)
        # nel_y = ceil((H_gap/2)/h_elem)

        # Generate mesh (use gmsh to ensure points at roller contact points)
        def generate_roller_mesh(h):
            # Number of elements
            nel_y = ceil((H_gap / 2) / h)
            nel_x_up = ceil(L_up / h)
            nel_x_roll = ceil(l_r / h)
            nel_x_down = ceil(L_down / h)

            # Mesh geometry
            mesh_size = 0.1
            mesh_name = "roller_mesh"
            gmsh.initialize()
            gmsh.model.add(mesh_name)
            # Points
            p1 = gmsh.model.geo.addPoint(
                -(L_up + l_r), 0, 0, mesh_size
            )  # bottom points
            p2 = gmsh.model.geo.addPoint(-l_r, 0, 0, mesh_size)
            p3 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
            p4 = gmsh.model.geo.addPoint(L_down, 0, 0, mesh_size)
            p5 = gmsh.model.geo.addPoint(
                -(L_up + l_r), H_gap / 2, 0, mesh_size
            )  # top points
            p6 = gmsh.model.geo.addPoint(-l_r, H_gap / 2, 0, mesh_size)
            p7 = gmsh.model.geo.addPoint(0, H_gap / 2, 0, mesh_size)
            p8 = gmsh.model.geo.addPoint(L_down, H_gap / 2, 0, mesh_size)
            # Lines
            c1 = gmsh.model.geo.addLine(p1, p2)  # bottom horiz lines
            c2 = gmsh.model.geo.addLine(p2, p3)
            c3 = gmsh.model.geo.addLine(p3, p4)
            c4 = gmsh.model.geo.addLine(p1, p5)  # vert Lines
            c5 = gmsh.model.geo.addLine(p4, p8)
            c6 = gmsh.model.geo.addLine(p5, p6)  # upper horiz lines
            c7 = gmsh.model.geo.addLine(p6, p7)
            c8 = gmsh.model.geo.addLine(p7, p8)
            # Surfaces
            cl1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c5, -c8, -c7, -c6, -c4])
            s1 = gmsh.model.geo.addPlaneSurface([cl1])
            # Meshing setup
            gmsh.model.geo.mesh.setTransfiniteSurface(s1, cornerTags=[p1, p4, p8, p5])
            gmsh.model.geo.mesh.setTransfiniteCurve(c1, nel_x_up + 1, coef=1)
            gmsh.model.geo.mesh.setTransfiniteCurve(c2, nel_x_roll + 1, coef=1)
            gmsh.model.geo.mesh.setTransfiniteCurve(c3, nel_x_down + 1, coef=1)
            gmsh.model.geo.mesh.setTransfiniteCurve(c5, nel_y + 1, coef=1)
            gmsh.model.geo.mesh.setTransfiniteCurve(c8, nel_x_down + 1, coef=1)
            gmsh.model.geo.mesh.setTransfiniteCurve(c7, nel_x_roll + 1, coef=1)
            gmsh.model.geo.mesh.setTransfiniteCurve(c6, nel_x_up + 1, coef=1)
            gmsh.model.geo.mesh.setTransfiniteCurve(c4, nel_y + 1, coef=1)

            # Mesh and write to file
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            gmsh.write(mesh_name + ".msh")
            # Import gmsh mesh - first write msh to xdmf, then import to fenics
            if comm_rank == 0:
                msh = meshio.read(mesh_name + ".msh")
                msh.prune_z_0()
                meshio.write(
                    mesh_name + ".xdmf",
                    meshio.Mesh(
                        points=msh.points,
                        cells={"triangle": msh.get_cells_type("triangle")},
                    ),
                )  # add cell data if physical groups specified
            mesh = Mesh()
            with XDMFFile(mesh_name + ".xdmf") as infile:
                infile.read(mesh)
            if comm_rank == 0:
                os.remove(mesh_name + ".msh")
                os.remove(mesh_name + ".xdmf")
                os.remove(mesh_name + ".h5")

            return mesh

        # mesh = generate_roller_mesh(h_elem)
        mesh = RectangleMesh(
            Point(-(L_up + l_r), 0),
            Point(L_down, H_gap / 2),
            ceil((L_up + l_r + L_down) / h_elem),
            ceil((H_gap / 2) / h_elem),
            "right",
        )
        dx = Measure("dx", domain=mesh)

        # Define boundaries
        left = CompiledSubDomain(
            "on_boundary && near(x[0], side, tol)", side=-(L_up + l_r), tol=1e-7
        )
        bot = CompiledSubDomain(
            "on_boundary && near(x[1], side, tol)", side=0.0, tol=1e-7
        )
        right = CompiledSubDomain(
            "on_boundary && near(x[0], side, tol)", side=L_down, tol=1e-7
        )
        top_down = CompiledSubDomain(
            "on_boundary && near(x[1], side, tol) && x[0]>=x2-tol",
            side=H_gap / 2,
            x2=0,
            tol=1e-7,
        )
        top_roll = CompiledSubDomain(
            "on_boundary && near(x[1], side, tol) && x[0]>=x1-tol && x[0]<=x2+tol",
            side=H_gap / 2,
            x1=-l_r,
            x2=0,
            tol=1e-7,
        )
        top_up = CompiledSubDomain(
            "on_boundary && near(x[1], side, tol) && x[0]<=x1+tol",
            side=H_gap / 2,
            x1=-l_r,
            tol=1e-7,
        )
        # Mark boundaries
        facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        facets.set_all(0)
        left.mark(facets, 1)
        bot.mark(facets, 2)
        right.mark(facets, 3)
        top_down.mark(facets, 4)
        top_roll.mark(facets, 5)
        top_up.mark(facets, 6)
        ds = Measure("ds", domain=mesh, subdomain_data=facets)  # partition boundary
        # ds = ds(subdomain_data=facets)

        # Function spaces for thermo-chemical problem
        V_t = FunctionSpace(mesh, "CG", 1)
        w = TestFunction(V_t)
        T = TrialFunction(V_t)
        T_ = Function(V_t, name="T_[K]")
        T_C = Function(V_t, name="T_[C]")
        T_noFP = Function(V_t, name="T_noFP")
        theta_ = Function(V_t, name="theta")
        V_a = FunctionSpace(mesh, "CG", 1)
        beta = TestFunction(V_a)
        dal = TrialFunction(V_a)
        al = Function(V_a, name="alpha")
        al_last = Function(V_a)  # alpha at previous iteration
        al_diff = Function(V_a)  # difference in alpha from previous iteration

        # Boundary conditions
        T_roller = Expression("T", degree=1, T=T_r)
        # bc_roller = DirichletBC(V_t, T_roller, facets, 5) # temperature at roller
        T_in = Expression("T", degree=1, T=T0)
        bc_T_in = DirichletBC(V_t, T_in, facets, 1)  # initial temperature of tow
        # bcs_T_roller = [bc_roller, bc_T_in]
        bcs_T = [bc_T_in]

        alpha_in = Expression("a", degree=1, a=alpha0)
        bc_a_in = DirichletBC(V_a, alpha_in, facets, 1)  # initial degree of cure of tow
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

        v_ = Expression(("v", "0."), degree=1, v=V_r)  # Velocity field

        # phi = Expression() # fiber volume fraction field
        class FiberVolFrac(UserExpression):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def eval(self, values, x):
                if x[0] <= -l_r:
                    phi = phi_1
                elif x[0] <= 0:
                    phi = (
                        H_tow
                        * phi_1
                        / (H_tow - 2 * (sqrt(R_r**2 - x[0] ** 2) - R_r * cos(a_r)))
                    )
                else:
                    phi = phi_2
                values[0] = phi

            def value_shape(self):
                return ()

        phi = FiberVolFrac()

        # Homogenized properties
        k_par = k_par_fun(phi)  # thermal cond. parallel to fibers (x axis)
        k_perp = k_perp_fun(phi)  # thermal cond. perpendiculat to fibers (y axis)
        k_tens = as_tensor([[k_par, 0.0], [0.0, k_perp]])  # thermal conductivity tensor
        rho_bar = rho_bar_fun(phi)  # homogenized density
        Cp_bar = Cp_bar_fun(phi)  # homogenized specific heat
        rho_Cp_bar = rho_Cp_bar_fun(
            phi
        )  # product of homogenized density and specific heat

        # Variational form for degree of cure problem - NONDIMENSIONAL
        tau_a = stablz / V_r
        g = (1 - al) ** n_ * al**m_ / (1 + exp(Ca * (al - alpha_c)))
        F_a = (
            (beta + tau_a * dot(v_, grad(beta)))
            * (dot(v_, grad(al)) - A_ * exp(-Er / R_ / T_) * g)
            * dx
        )
        tang_a = derivative(F_a, al, dal)

        # Variational form for thermal problem
        Pe = (
            V_r * stablz / 2 / (k_par / rho_Cp_bar)
        )  # Peclet number for thermal problem (diffusivity in x direction)
        tau_T = stablz / V_r * (1 / tanh_ufl(Pe) - 1 / Pe)
        # ONLY APPLYING SUPG TEST FN TO MATERIAL DERIVATIVE TERMS
        F_T = (
            dot(grad(w), dot(k_tens, grad(T))) * dx
            + (w + tau_T * dot(v_, grad(w)))
            * (
                rho_Cp_bar * dot(v_, grad(T))
                - rhor * Hr * (1 - phi) * dot(v_, grad(al))
            )
            * dx
            + w * 1 / R_tc * (T - T_roller) * ds(5)
        )  # + w*h_conv*(T - Tamb)*ds(4)
        F_T_adiab = (
            dot(grad(w), dot(k_tens, grad(T))) * dx
            + (w + tau_T * dot(v_, grad(w)))
            * (
                rho_Cp_bar * dot(v_, grad(T))
                - rhor * Hr * (1 - phi) * dot(v_, grad(al))
            )
            * dx
        )  # + w*h_conv*(T - Tamb)*ds(4)
        F_T_noFP = (
            dot(grad(w), dot(k_tens, grad(T))) * dx
            + (w + tau_T * dot(v_, grad(w))) * (rho_Cp_bar * dot(v_, grad(T))) * dx
            + w * 1 / R_tc * (T - T_roller) * ds(5)
        )  # + w*h_conv*(T - Tamb)*ds(4)

        # Solver for degree of cure
        # problem_alpha = LinearVariationalProblem(lhs(F_a), rhs(F_a), al_, bcs_a, form_compiler_parameters=ffc_options)
        # solver_alpha = LinearVariationalSolver(problem_alpha)
        problem_alpha = NonlinearVariationalProblem(
            F_a, al, bcs=bcs_a, J=tang_a, form_compiler_parameters=ffc_options
        )
        solver_alpha = NonlinearVariationalSolver(problem_alpha)
        prms_a = solver_alpha.parameters
        prms_a["nonlinear_solver"] = "newton"
        prms_a["newton_solver"]["report"] = True
        prms_a["newton_solver"]["absolute_tolerance"] = 1e-17
        prms_a["newton_solver"]["relative_tolerance"] = 1e-14
        prms_a["newton_solver"]["error_on_nonconvergence"] = False
        prms_a["newton_solver"]["convergence_criterion"] = "residual"
        prms_a["newton_solver"]["maximum_iterations"] = 8
        prms_a["newton_solver"]["relaxation_parameter"] = 1.0
        # prms_a["newton_solver"]["linear_solver"] = "gmres"
        # prms_a["newton_solver"]["preconditioner"] = "hypre_amg"
        # prms_a["newton_solver"]['krylov_solver']['absolute_tolerance'] = 1e-10#1e-8#1e-12
        # prms_a["newton_solver"]['krylov_solver']['relative_tolerance'] = 1e-10#1e-8#1e-12
        # prms_a["newton_solver"]['krylov_solver']['maximum_iterations'] = 1000
        # prms_a["newton_solver"]['krylov_solver']['monitor_convergence'] = False

        # Solver for temperature - with roller boundary conditions
        problem_temp_roller = LinearVariationalProblem(
            lhs(F_T), rhs(F_T), T_, bcs_T, form_compiler_parameters=ffc_options
        )
        solver_temp_roller = LinearVariationalSolver(problem_temp_roller)
        prms_T = solver_temp_roller.parameters
        prms_T["linear_solver"] = "gmres"
        prms_T["preconditioner"] = "hypre_amg"
        prms_T["krylov_solver"]["absolute_tolerance"] = 1e-12
        prms_T["krylov_solver"]["relative_tolerance"] = 1e-12
        prms_T["krylov_solver"]["maximum_iterations"] = 100

        # Solver for temperature - with adiabatic boundary at rollers
        problem_temp_adiab = LinearVariationalProblem(
            lhs(F_T_adiab),
            rhs(F_T_adiab),
            T_,
            bcs_T,
            form_compiler_parameters=ffc_options,
        )
        solver_temp_adiab = LinearVariationalSolver(problem_temp_adiab)
        prms_T = solver_temp_adiab.parameters
        prms_T["linear_solver"] = "gmres"
        prms_T["preconditioner"] = "hypre_amg"
        prms_T["krylov_solver"]["absolute_tolerance"] = 1e-12
        prms_T["krylov_solver"]["relative_tolerance"] = 1e-12
        prms_T["krylov_solver"]["maximum_iterations"] = 100

        # Solver for temperature - WITHOUT FP
        problem_temp_roller_noFP = LinearVariationalProblem(
            lhs(F_T_noFP),
            rhs(F_T_noFP),
            T_noFP,
            bcs_T,
            form_compiler_parameters=ffc_options,
        )
        solver_temp_roller_noFP = LinearVariationalSolver(problem_temp_roller_noFP)
        prms_T = solver_temp_roller_noFP.parameters
        prms_T["linear_solver"] = "gmres"
        prms_T["preconditioner"] = "hypre_amg"
        prms_T["krylov_solver"]["absolute_tolerance"] = 1e-12
        prms_T["krylov_solver"]["relative_tolerance"] = 1e-12
        prms_T["krylov_solver"]["maximum_iterations"] = 100

        ################################### SOLUTION #################################################

        # Output file
        file_results = XDMFFile(
            "results_comp_print_tcr_V_r_" + str(V_r) + "_Rtc_" + str(R_tc) + ".xdmf"
        )
        # file_results = XDMFFile("results_comp_print_{}.xdmf".format(write_str))
        file_results.parameters["flush_output"] = True
        file_results.parameters["functions_share_mesh"] = True

        file_results.write(T_, 0)
        file_results.write(al, 0)
        file_results.write(theta_, 0)

        # Write functions for parameters that vary with fiber volume fraction (write the ratios, tilde parameters)
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
        alpha_dof = np.array(V_a.dofmap().dofs())  # dofs for alpha among all processors
        alpha_midline_dofs_local = np.where(alpha_coords[:, 1] <= 0)[
            0
        ]  # midline dofs, using this core's dofs
        alpha_midline_dofs = alpha_dof[
            alpha_midline_dofs_local
        ]  # midline dofs, using global dofs

        T_roller_list = []
        alpha_max_list = []
        front_loc_list = []
        Q_roller_list = []
        Q_roller_noFP_list = []

        load_prev = 0
        # Iterative solution for steady problem
        for l_iter, load in enumerate(loads):
            print_parallel("Solving for load {}".format(load))

            # Update bounday condition for load
            T_roller.T = T0 + load * (T_r - T0)  # ramp roller temperature

            for k in range(1, max_iter + 1):
                solver_temp_roller.solve()

                try:
                    (it_nl, conv_nl) = solver_alpha.solve()

                except:
                    conv_nl = 0

                # If solution for alpha does not converge, use alternative solver
                if conv_nl != 1:
                    print_parallel("Not converged")

                    al.assign(al_last)

                    x_ = Function(V_a)  # solution vector
                    tol_abs_nl = 1e-12
                    tol_rel_nl = 1e-4  # checks if change from last iteration is small
                    tol_inc_nl = 1e-13
                    max_iter_nl = 8
                    bcs_a_hold = [DirichletBC(V_a, Constant(0.0), facets, 1)]

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

                    while (
                        error > tol_abs_nl
                        and error_rel > tol_rel_nl
                        and error_inc > tol_inc_nl
                        and it <= max_iter_nl
                    ):
                        it += 1
                        A, b = assemble_system(tang_a, -F_a, bcs_a_hold)
                        solve(A, x_.vector(), b)
                        error = b.norm("l2")
                        error_rel = (
                            np.abs(error - error_last) / error_last
                        )  # if it > 1 else 1
                        error_inc = norm(x_)
                        error_last = error
                        print_parallel(
                            "Iteration: {:d}; Residual: {:.3e}; Relative: {:.3e}, Incremental: {:.3e} ".format(
                                it, error, error_rel, error_inc
                            )
                        )
                        al.vector()[:] += x_.vector()[:]

                        # file_results.write(al, load_prev + (load-load_prev)*(k-1 + it/max_iter_nl)/max_iter)

                        al.vector()[:] = np.maximum(al.vector()[:], alpha0 - 1e-4)
                        al.vector()[:] = np.minimum(al.vector()[:], 1 - 1e-4)

                al.vector()[:] = np.minimum(al.vector()[:], 0.999)
                al.vector()[:] = np.maximum(al.vector()[:], alpha0 - 1e-4)

                al_diff.vector()[:] = al.vector()[:] - al_last.vector()[:]
                it_norm = norm(al_diff)
                print_parallel(
                    "\t Iteration {}, residual {} (tolerance {})".format(
                        k, it_norm, iter_tol
                    )
                )

                al_last.assign(al)

                # Write solution for current iteration
                # file_results.write(T_, load_prev + (load-load_prev)*k/max_iter)
                # file_results.write(al, load_prev + (load-load_prev)*k/max_iter)

                # Check if solution converge on this iteration
                if it_norm <= iter_tol:
                    print_parallel("\t Converged for iteration {}".format(k))
                    break

            # Get midline alpha values, sorted from left (x=-(L_up+l_r)) to right (x=L_down)
            x_coord_gather = comm.gather(
                alpha_coords[alpha_midline_dofs_local, 0], root=0
            )
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

            alpha_max = np.max(alpha_midline)
            try:
                front_loc = interp1d(alpha_midline, x_midline)(0.5)
            except:
                front_loc = -np.inf

            T_roller_list.append(T0 + load * (T_r - T0))
            alpha_max_list.append(alpha_max)
            front_loc_list.append(front_loc)

            theta_.assign(project((T_ - T0) / (T_max - T0), V_t))

            # Compute energy exchange with roller (power per unit thickness, from ONE roller)
            q_roller = assemble(-dot(dot(k_tens, grad(T_)), n) * ds(5))
            Q_roller_list.append(q_roller)

            # Compute energy WITH NO FP
            solver_temp_roller_noFP.solve()
            q_roller_noFP = assemble(-dot(dot(k_tens, grad(T_noFP)), n) * ds(5))
            Q_roller_noFP_list.append(q_roller_noFP)

            # Write converged solution for load step
            file_results.write(T_, load)
            file_results.write(al, load)
            file_results.write(theta_, load)

            file_results.write(T_noFP, load)

            load_prev = load

        # Now, remove heat from roller and solve again
        if V_r >= 3e-3:  # no need to run this if V_r >> V_front
            solve_noRoller = False

        if solve_noRoller:
            print_parallel("Solving adiabatic problem (roller contact removed)")
            for k in range(1, max_iter + 1):
                solver_temp_adiab.solve()
                (it_nl, conv_nl) = solver_alpha.solve()

                al.vector()[:] = np.minimum(al.vector()[:], 0.999)
                al.vector()[:] = np.maximum(al.vector()[:], alpha0 - 1e-4)

                al_diff.vector()[:] = al.vector()[:] - al_last.vector()[:]
                it_norm = norm(al_diff)
                print_parallel(
                    "\t Iteration {}, residual {} (tolerance {})".format(
                        k, it_norm, iter_tol
                    )
                )

                al_last.assign(al)

                theta_.assign(project((T_ - T0) / (T_max - T0), V_t))
                # file_results.write(T_, 1 + k/max_iter)
                # file_results.write(al, 1 + k/max_iter)
                # file_results.write(theta_, 1 + k/max_iter)

                # Check if solution converge on this iteration
                if it_norm <= iter_tol:
                    print_parallel("\t Converged for iteration {}".format(k))
                    break

            file_results.write(T_, 2)
            file_results.write(al, 2)
            file_results.write(theta_, 2)

        # Printout
        # print_file = open("data_comp_print_{}.csv".format(write_str),'w')
        print_parallel("")
        print_parallel("Roller contact length: {} m".format(l_r))

        print_parallel(
            "Load no\tT_r [C]\talpha_max\tfront_x\tQ_roller (from both rollers, power per unit thickness [W/m])\tQ_roller_noFP (from both rollers power per unit thickness [W/m])"
        )

        if comm_rank == 0 and (i, j) == (0, 0):
            print_file = open("data_comp_print.csv", "a+")
            print_file.write("Roller contact length [m],{}\n".format(l_r))
            print_file.write(
                "Load no,T_r [C],alpha_max,front_x,front_v (mm/s),Q_roller (power per unit thickness from both rollers [W/m]),Q_roller_noFP (from both rollers power per unit thickness [W/m]),R_tc,V_r,T0, Tamb, Hgap, phi1, phi2, alpha0, h_conv\n"
            )
            print_file.flush()

        for i, (
            T_r_i,
            alpha_max_i,
            front_loc_i,
            q_roller_i,
            q_roller_noFP_i,
        ) in enumerate(
            zip(
                T_roller_list,
                alpha_max_list,
                front_loc_list,
                Q_roller_list,
                Q_roller_noFP_list,
            )
        ):
            print_parallel(
                "{}\t{}\t{}\t{}\t{}\t{}".format(
                    i,
                    T_r_i - 273.15,
                    alpha_max_i,
                    front_loc_i,
                    2 * q_roller_i,
                    2 * q_roller_noFP_i,
                )
            )
            if comm_rank == 0:
                print_file.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        i,
                        T_r_i - 273.15,
                        alpha_max_i,
                        front_loc_i,
                        1.5,
                        2 * q_roller_i,
                        2 * q_roller_noFP_i,
                        R_tc,
                        V_r,
                        T0 - 273.15,
                        Tamb - 273.15,
                        H_gap,
                        phi_1,
                        phi_2_target,
                        alpha0,
                        h_conv,
                    )
                )
                print_file.flush()
        if comm_rank == 0:
            print_file.write("\n")
            print_file.flush()

if comm_rank == 0:
    print_file.close()
