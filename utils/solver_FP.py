# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
# 
# This code is developed for the solution of 1D frontal polymerization problems 
# with a coupled set of thermo-chemical equations. For more information, please
# refer to the following papers:


# Import statements
import os
import time
import math
import numpy as np

from dolfin import *
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning


# Set communicator (in case running in parallel)
comm = MPI.comm_world
comm_rank = MPI.rank(comm)

class FP_solver():
    """
    A solver class for 1D thermo-chemical problems.

    """

    QUAD_DEG = 2 # degree of quadrature

    def __init__(self, k, rho, Cp, H, A, E, kinetics_model, T0, alpha0, T_trig=None, t_trig=1, L=0.015, dt=1e-3, h_x=1e-5):
        """
        Initializes the FP solvers using the supplied parameters
        
        Parameters
        ----------
        k : float
            Thermal conductivity of monomer
        rho : float
            Density of monomer
        Cp : float
            Specific heat of monomer
        H : float
            Enthalpy of reaction
        A : float
            Pre-exponential factor of cure kinetics
        E : float
            Activation energy of reaction
        kinetics_model : a Kinetics_Model object
            A subclass of the Kinetics_Model class
        T0 : float
            Initial temperature of system (in deg. C)
        alpha0 : float
            Initial degree of cure of system
        T_trig : float
            The trigger temperature in deg. C. If no value supplied, uses T_max of adiabatic system
        t_trig : float
            Duration of trigger
        L : float (default value of 1.5 cm)
            Length of domain
        dt : float (default value of 1 ms)
            Time step for simulation
        h_x : float (default value of 1e-5 m)
            Element size
        """

        # Set thermo-chemical properties
        self.R_ = 8.314
        self.k =  k
        self.rho = rho
        self.Cp = Cp
        self.H = H
        self.A = A
        self.E = E
        
        self.kinetics_model = kinetics_model

        # Initial values
        self.T0 = T0 + 273.15 # convert to K
        self.alpha0 = alpha0 

        self.T_max = self.T0 + H/Cp*(1-alpha0) # max temperature of reaction in adiabatic system
        if not T_trig:
            self.T_trig = self.T_max
        else:
            self.T_trig = T_trig + 273.15
        self.t_trig = t_trig # time for trigger
        
        # Mesh and time-stepping parameters
        self.L = L
        self.nel_x = math.ceil(L/h_x) 
        self.delta_t = dt #5e-3

        self.ffc_options = self.setSystemParameters(self.QUAD_DEG)
        self.setupSolver() # initialize the solvers

    @staticmethod
    def printParallel(str):
        """ Only print on one core
        """
        if comm_rank == 0:
            print(str)

    @staticmethod
    def setSystemParameters(deg):
        """
        Set system parameters for fenics. Return the options for FFC
        
        Parameters
        ----------
        deg - int
            Degree of quadrature
        """
        
        set_log_level(30) # (Error level=40, warning level=30)
                          # this is set to prevent output while solving
        parameters["linear_algebra_backend"] = "PETSc"
        parameters["form_compiler"]["cpp_optimize"] = True
        ffc_options = {"optimize": True, \
                    "eliminate_zeros": True, \
                    "precompute_basis_const": True, \
                    "precompute_ip_const": True}

        parameters["form_compiler"]["representation"] = 'quadrature'  #this is deprecated
        # The following shuts off a deprecation warning for quadrature representation:
        warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

        parameters["form_compiler"]["quadrature_degree"] = deg

        return ffc_options

    @staticmethod
    def local_project(v, V, u):
        """
        Perform an element-wise projection of values in expression v
        onto function u on function space V

        Parameters
        ----------
        v - ufl operator
            A ufl algebraic expression referencing the data to be projected
        V - dolfin function space
            The function space to which function u belongs
        u - dolfin function
            The function onto which values are projected
        """

        parameters["form_compiler"]["representation"] = 'quadrature'
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dx
        b_proj = inner(v, v_)*dx
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        solver.solve_local_rhs(u)


    def setupSolver(self):
        """
        Initialize the solvers needed to solve the 1D FP problem
        """

        # Create a 1D mesh of length L with number of divisions nel_x
        mesh = IntervalMesh(comm, self.nel_x, 0.0, self.L)

        # Define the left and right boundaries of domain
        left =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 0.0, tol=1e-7)
        right =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = self.L, tol=1e-7)
        facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        facets.set_all(0)
        left.mark(facets, 1)
        right.mark(facets, 2)
        ds = Measure('ds', domain=mesh, subdomain_data=facets) # partition boundary into left and right

        # Create function spaces for temperature and degree of cure fields
        V = FunctionSpace(mesh, "CG", 1) # continuous 1st order polynomials for temperature

        dx = Measure('dx', domain=mesh, metadata={"quadrature_degree":self.QUAD_DEG, "quadrature_scheme": "default"})
        Qelement = FiniteElement("Quadrature", mesh.ufl_cell(), degree=self.QUAD_DEG, quad_scheme="default")
        Q = FunctionSpace(mesh, Qelement) # define degree of cure at quadrature points

        # Create functions
        w = TestFunction(V)                     # test function for temperature
        T = TrialFunction(V)                    # trial function for temperature
        T_ = Function(V, name="Temperature")    # function storing values of temperature at current time
        T_n = Function(V)                       # function storing values of temperature at previous time
        T_max = Function(V, name="T_max")       # function tracking maximum temperature in domain over time
        alpha_ = Function(Q)                    # function storing values of degree of cure at current time
        alpha_n = Function(Q)                   # function storing values of degree of cure at previous time
        alpha_plot = Function(V, name="alpha")  # function used for displaying degree of cure in results file
        # Y2 = FunctionSpace(mesh, "DG", 0) 
        # alpha_plot = Function(Y2, name="alpha")

        # Define boundary conditions
        T_trig_val = Constant(self.T_trig)
        bc_trig = DirichletBC(V, T_trig_val, left) # dirichlet boundary condition on left end of domain
        bcs_trig = [bc_trig] # boundary conditions with thermal trigger
        bcs_adiab = []    # boundary conditions after thermal trigger removed (adiabatic)

        # Apply initial temperature and degree of cure
        T_init = Constant(self.T0)
        T_.interpolate(T_init)
        T_n.interpolate(T_init)
        alpha_init = Constant(self.alpha0)
        alpha_.interpolate(alpha_init)
        alpha_n.interpolate(alpha_init)

        # Define variational problem for solving for temperature
        dt = Constant(self.delta_t)
        F_T = self.rho*self.Cp*w*T*dx - self.rho*self.Cp*w*T_n*dx + dt*self.k*dot(grad(w),grad(T))*dx \
                - self.rho*self.H*w*alpha_*dx + self.rho*self.H*w*alpha_n*dx
                
        # Define solver for problem with thermal trigger
        problem_T_trig = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_trig, form_compiler_parameters=self.ffc_options)
        self.solver_T_trig = LinearVariationalSolver(problem_T_trig)
        self.solver_T_trig.parameters['linear_solver'] = 'cg'
        self.solver_T_trig.parameters['preconditioner'] = 'hypre_euclid'
        # self.solver_T_trig.parameters['krylov_solver']['absolute_tolerance'] = 1e-8
        # self.solver_T_trig.parameters['krylov_solver']['relative_tolerance'] = 1e-10
        # self.solver_T_trig.parameters['krylov_solver']['maximum_iterations'] = 200

        # Define solve for adiabatic problem after thermal trigger removed
        problem_T_adiab = LinearVariationalProblem(lhs(F_T), rhs(F_T), T_, bcs=bcs_adiab, form_compiler_parameters=self.ffc_options)
        self.solver_T_adiab = LinearVariationalSolver(problem_T_adiab)
        self.solver_T_adiab.parameters['linear_solver'] = 'cg'
        self.solver_T_adiab.parameters['preconditioner'] = 'hypre_euclid'
        # self.solver_T_adiab.parameters['krylov_solver']['absolute_tolerance'] = 1e-8
        # self.solver_T_adiab.parameters['krylov_solver']['relative_tolerance'] = 1e-10
        # self.solver_T_adiab.parameters['krylov_solver']['maximum_iterations'] = 200

        # Define the expression used for explicit update of degre of cure
        g = self.kinetics_model.eval(alpha_n)
        self.alpha_update = alpha_n + dt*self.A*exp(-self.E/(self.R_*T_n)) * g

        # Store variables for use outside function
        self.T_ = T_
        self.T_n = T_n
        self.T_max = T_max
        self.alpha_ = alpha_
        self.alpha_n = alpha_n
        self.alpha_plot = alpha_plot


    def solve(self, t_end, outputFreq_t=1, outputFreq_x=1):
        """
        Solve the thermo-chemical equations

        Parameters
        ----------
        t_end - float
            The time span of the simulation
        output_t - float (default value of 1, ie. save at every time step)
            Frequency in time steps at which to save data
        output_x - int (default value of 1, ie. save every grid point)
            Spacing between sampling nodal points at which to save data

        Returns
        -------
        x_data, t_data, T_data, alpha_data - 
            Arrays of shape (num_t, num_x), where num_t is the number of sample points in time
            and num_x is the number of sample points in space at which to store data, containing
            the x coordinate, time, temperature and degree of cure, respectively.
        """

        t = 0                                           # current time
        step = 1                                        # time step counter
        t_idx = 1                                       # counter for saving at sampled time points
        solve_complete = False                          # boolean for whether simulation is complete
        num_steps = math.ceil(t_end / self.delta_t)     # number of time steps in simulation

        # Initialize x coordinate and time vectors at sampled points
        save_x_idx = np.arange(0, self.nel_x, outputFreq_x)
        print(save_x_idx.shape)                 # indices of dof at which to save data
        save_x_idx[-1] = self.nel_x                                         # ensure end point is included in sampling
        coords = self.T_.function_space().tabulate_dof_coordinates()
        coords_sort_idx = coords.reshape(-1).argsort()                      # get the sorted indices for dof in positive x direction
        x_coords = coords[coords_sort_idx][save_x_idx]                      # vector containing the x coordinates at sampled points
        t_arr = np.linspace(0, t_end, math.ceil(num_steps/outputFreq_t)+1)  # vector containing the t values at sampled times

        # Initialize arrays to store temperature and degree of cure over time at sampled
        x_data, t_data = np.meshgrid(x_coords, t_arr)
        T_data = np.full(x_data.shape, self.T0)
        alpha_data = np.full(x_data.shape, self.alpha0)
        T_peak = np.full(x_data.shape, self.T0)

        idx_end = math.ceil(0.9*self.nel_x + 1) # dof index beyond which front not recorded (does not save once front becomes close to end of domain)
        
        # Simulation loop
        start_time = time.time()
        while not solve_complete and step < num_steps+1:
            t += self.delta_t

            # Solve for degree of cure, restricting to range alpha0 - 1
            self.local_project(self.alpha_update, self.alpha_.function_space(), self.alpha_)
            self.alpha_.vector()[:] = np.minimum(self.alpha_.vector()[:],1-1e-4)
            self.alpha_.vector()[:] = np.maximum(self.alpha_.vector()[:],self.alpha0-1e-4)

            # Solver for temperature, with trigger applied until t_trig
            if t < self.t_trig:
                self.solver_T_trig.solve()
            else:
                self.solver_T_adiab.solve()

            # Save results at specified frequency
            if step % outputFreq_t == 0:
                
                # self.printParallel("Saving results for t = {} s, elapsed simulation time = {} s".format(t, time.time()-start_time))
                self.printParallel(f"Saving results for t = %f s, Elapsed simulation time: %f s" % (t, time.time() - start_time))

                # Project alpha from quadrature points to nodes
                self.local_project(self.alpha_, self.alpha_plot.function_space(), self.alpha_plot)
                self.alpha_plot.vector()[:] = np.minimum(self.alpha_plot.vector()[:],1-1e-4)
                self.alpha_plot.vector()[:] = np.maximum(self.alpha_plot.vector()[:],self.alpha0-1e-4)

                alpha_sorted = self.alpha_plot.vector()[coords_sort_idx] # sort alpha values along positive x
                # Save temperature and degree of cure
                if alpha_sorted[idx_end] < 0.5:
                    T_data[t_idx, :] = self.T_.vector()[coords_sort_idx][save_x_idx]
                    alpha_data[t_idx, :] = alpha_sorted[save_x_idx]
                    t_idx = t_idx + 1
                else:
                    solve_complete = True

            # Update parameters for next iteration
            self.T_n.assign(self.T_)
            self.alpha_n.assign(self.alpha_)
            step += 1

        # Remove data points for when front was close to end of domain
        x_data = x_data[0:t_idx, :]
        t_data = t_data[0:t_idx, :]
        T_data = T_data[0:t_idx, :]
        alpha_data = alpha_data[0:t_idx, :]

        return x_data, t_data, T_data, alpha_data
        
