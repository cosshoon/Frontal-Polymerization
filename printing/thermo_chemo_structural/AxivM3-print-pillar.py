from dolfin import *
import time
import numpy as np

###################################################
set_log_level(40)  #Error level=40, warning level=30
parameters["linear_algebra_backend"] = "PETSc"
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
###################################################

######the following needs to be added to use quadrature elements
parameters["form_compiler"]["representation"] = 'quadrature'  #this is deprecated
# The following shuts off a deprecation warning for quadrature representation:
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
#################################################################################

#Function definitions############################################################

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

#For local projections
def local_project_2(v, V):
	parameters["form_compiler"]["representation"] = 'quadrature'
	dv = TrialFunction(V)
	v_ = TestFunction(V)
	a_proj = inner(dv, v_) * dx
	b_proj = inner(v, v_) * dx
	solver = LocalSolver(a_proj, b_proj)
	solver.factorize()
	u = Function(V)
	solver.solve_local_rhs(u)
	return u

#Calculates the strain energy corresponding to the spring in equilibrium branch
def equilbriumbranch(u,Fte,Fcs,iFr,iF33r,detiFr):
	F=I + grad(u)
	Fer=F*iFr/Fte/Fcs
	C = Fer.T*Fer   	
	F33er=(1+u[0]/xm[0])*iF33r/Fte/Fcs
	Ic = tr(C)+F33er*F33er
	J = det(F)*detiFr*F33er
	psi_eq=(mu_Eq/2)*(Ic - 3)- (mu_Eq)*ln(J) + (lmbda_Eq/2)*(J-1)**2
	
	return psi_eq

#Calculates the strain energy corresponding to the spring in non-equilibrium branch
def nonequilbriumbranch(u,Fte,Fcs,Cv,Cvinv,Cv33,mu_Neq,lmbda_Neq):
	F=I + grad(u)
	FeNeq=F/Fte/Fcs
	CNeq = FeNeq.T*FeNeq 
	F33Neq=(1+u[0]/xm[0])/Fte/Fcs
	I1ep = inner(CNeq,Cvinv)+(F33Neq**2)/Cv33
	J1e  = det(F)/(Fte*Fte*Fcs*Fcs)*F33Neq/sqrt(det(Cv)*Cv33)
	psi_neq=(mu_Neq/2)*(I1ep - 3)- (mu_Neq)*ln(J1e) + (lmbda_Neq/2)*(J1e-1)**2
	return psi_neq


#Solve the balance of forces for R_u=0
def implicitgradientflow(R_u, Jac_u, Jac3, bcs_du0, bcs_du, terminate, terminate2, omega ):

	##iterations##
	nIter = 0
	rnorm = 1000.0
	rnorm_prev=1000.0
		
	while nIter < 20: 	
		nIter += 1
		
		if nIter>10 and terminate2==0:
			terminate=1
			break
		
		
		if terminate2==1:
			if omega<omega0*1e-3:
				if nIter<5:
					solver_u.parameters["maximum_iterations"] = 500
				else:	
					solver_u.parameters["maximum_iterations"] = 150
			else:
				if nIter<15:
					solver_u.parameters["maximum_iterations"] = 450
				else:	
					solver_u.parameters["maximum_iterations"] = 250
		
		
		
		if nIter==1:
			Amatrix, b = assemble_system(Jac_u+(1/omega)*Jac3, -R_u, bcs_du0)  #
		else:
			Amatrix, b = assemble_system(Jac_u+(1/omega)*Jac3, -R_u, bcs_du)   #
		
		rnorm=b.norm('l2')
		
		if comm_rank==0:	
			print('Iteration number= %d' %nIter,  'Residual= %e' %rnorm)
			
		if rnorm < rtol:             #residual check
			break
		
		if rnorm > rnorm_prev*10 or np.isnan(rnorm) is True:          
			terminate=1
			break
		
		if nIter==19 and rnorm>1e-1:
			terminate=1
			break
		
		
		rnorm_prev=rnorm
		
		converged = solver_u.solve(Amatrix, u_inc.vector(), b);
		u.vector().axpy(1, u_inc.vector())
	
	return u, terminate

#Calculate right hand side of evolution equation called Ge
def Ge(u,Cv,Cv33, Ft, Fc, mu_Neq, vis):
	F = (Identity(len(u)) + grad(u))/Ft/Fc  #replace F with F/Fte/Fcs
	F33=(1+u[0]/xm[0])/Ft/Fc  #replace F33 with F33/Fte/Fcs
	C = F.T * F
	CCinv = C * inv(Cv)
	CvinvC = inv(Cv) * C
	CCinv33=(F33**2)/Cv33
	Ie1 = local_project_2(tr(CCinv) + CCinv33, Y)
	Ie2h= local_project_2(1./ 2 * (inner(CvinvC, CCinv)+CCinv33**2), Y)
	G=local_project_2(mu_Neq/vis* (C - Ie1 / 3. * Cv), Vv)
	G33=local_project_2(mu_Neq/vis* (F33**2 - Ie1 / 3. * Cv33), Y)
	return G,G33

# Calculates the intermediate terms in the solution of evolution eq for Cv using a 5th order Runge-Kutta scheme
def k_terms(dt, u, uold, Cv_old, Cv33_old, Ft, Fc, mu_Neq, vis):

	u_quart = uold + 0.25 * (u - uold)
	u_half = uold + 0.5 * (u - uold)
	u_three_quart = uold + 0.75 * (u - uold)
	k1, k1_33 = Ge(uold, Cv_old, Cv33_old, Ft, Fc, mu_Neq, vis)
	k2, k2_33 = Ge(u_half, Cv_old + k1 * dt / 2, Cv33_old + k1_33 * dt / 2, Ft, Fc, mu_Neq, vis)
	k3, k3_33 = Ge(u_quart, Cv_old + 1. / 16 * dt * (3 * k1 + k2), Cv33_old + 1. / 16 * dt * (3 * k1_33 + k2_33), Ft, Fc, mu_Neq, vis)
	k4, k4_33 = Ge(u_half, Cv_old + dt / 2. * k3, Cv33_old + dt / 2. * k3_33, Ft, Fc, mu_Neq, vis)
	k5, k5_33 = Ge(u_three_quart, Cv_old + 3. / 16 * dt * (-k2 + 2. * k3 + 3. * k4), Cv33_old + 3. / 16 * dt * (-k2_33 + 2. * k3_33 + 3. * k4_33), Ft, Fc, mu_Neq, vis)
	k6, k6_33 = Ge(u, Cv_old + (k1 + 4. * k2 + 6. *k3 - 12. * k4 + 8. * k5 ) * dt / 7., Cv33_old + (k1_33 + 4. * k2_33 + 6. *k3_33 - 12. * k4_33 + 8. * k5_33 ) * dt / 7., Ft, Fc, mu_Neq, vis)

	kfinal = dt / 90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)
	kfinal=local_project_2(kfinal, Vv)
	kfinal33 = dt / 90 * (7 * k1_33 + 32 * k3_33 + 12 * k4_33 + 32 * k5_33 + 7 * k6_33)
	kfinal33=local_project_2(kfinal33, Y)
	return kfinal, kfinal33
	
#eigen_function refers to the function f defined in equation 12 in the paper on Surface patterning (JMPS 22)	
def eigen_function(a):
	k, ac=20, 0.4
	return k*exp(k*(a-ac))/(1+exp(k*(a-ac)))**2

#non-symmetric small strain tensor, in other words, epsilon_elastic= F-I
def epsilon_elastic(v):
	return (grad(v))

#epsilon_33= F33-1	
def epsilon_33(v):
	return v[0]/xm[0]	

#################################################################################################








#### Create mesh
x_right=0.00075
y_right=0.01
meshsize=2e-5
mesh=RectangleMesh(comm, Point(0.0,0.0), Point(x_right,y_right),int(x_right/meshsize),int(y_right/meshsize))
xm = SpatialCoordinate(mesh)
h=FacetArea(mesh)          #area/length of a cell facet on a given mesh
h_avg = (h('+') + h('-'))/2



#Variable and finite element function definintions start here

comm = MPI.comm_world 
comm_rank = MPI.rank(comm)


####Total time T and time step dt
T = 15.0            # final time
dt=0.001   #dt=0.0005 recommended when thermo-chemical instabilities
num_steps = int(T/dt)   

#Initial conditions and trigger
T_0=293.15
alpha_0=0.1
T_trig=483.0
t_trig=1.0


########### Define material properties##########################################################################################
kappa, rho, Cp, H, A, Er, R, n, m, Ca, alpha_c = 0.15, 980.0, 1600.0, 350000, 8.55e15, 110750.0, 8.314, 1.72, 0.77, 14.48, 0.41
at=2.5e-4 #coefficient of thermal expansion
cure_shrink= 0.07539  #cure shrinkage strain

#Viscosity parameters
vis1, vis2, vis3=73.4, 60, 15   

#Equilibrium branch elastic parameters before and after polymerization
E1, E2=0.09*(10**3), 3.5*(10**6) 
nu1, nu2= 0.45, 0.4

#Three non-equilibrium branch elastic parameters
mu_Neq1, mu_Neq2, mu_Neq3=0.73*(10**3), 0.066*(10**3), 0.003*(10**3)
lmbda_Neq1, lmbda_Neq2, lmbda_Neq3=mu_Neq1*10, mu_Neq2*10, mu_Neq3*10

#Heat loss coefficient and environmental temperature
hL=100
Tenv=373.15

#time after which material is extruded out=0.25s
t_add=0.25
addsteps=int(t_add/dt)
printsteps=addsteps*1

#Make the choice for Vflow and Vprint
Vflow=0.00085 
Vprint=0.00085
Ltime=1.0 #time after which reaction is triggered

# Gravitational force per unit volume
B  = Constant((0.0, -9614))  

#####################################################################################################

##### Sigma_nozzle obtained from the viscoelastic fluid model for a cylindrical nozzle with diameter=1.54mm
#1*0.00085
sigma_nozzle100= sym(as_tensor([[-1*(-3.2*pow(10,10)*pow(xm[0],3)-1.25*pow(10,8)*pow(xm[0],2)-2.06*pow(10,4)*xm[0]), 5.51*pow(10,5)*xm[0]],[5.51*pow(10,5)*xm[0], -1*(1.12*pow(10,11)*pow(xm[0],3)+1.04*pow(10,9)*pow(xm[0],2)+3.08*pow(10,5)*xm[0])]]))

#1.25*0.00085
sigma_nozzle125= sym(as_tensor([[-1*(-7.66*pow(10,10)*pow(xm[0],3)-1.43*pow(10,8)*pow(xm[0],2)-2.97*pow(10,4)*xm[0]), 6.38*pow(10,5)*xm[0]],[6.38*pow(10,5)*xm[0], -1*(3.16*pow(10,11)*pow(xm[0],3)+9.39*pow(10,8)*pow(xm[0],2)+4.62*pow(10,5)*xm[0])]]))

#1.5*0.00085
sigma_nozzle150= sym(as_tensor([[-1*(-1.3*pow(10,11)*pow(xm[0],3)-1.53*pow(10,8)*pow(xm[0],2)-3.86*pow(10,4)*xm[0]), 7.09*pow(10,5)*xm[0]],[7.09*pow(10,5)*xm[0], -1*(9.7*pow(10,11)*pow(xm[0],3)+6.55*pow(10,8)*pow(xm[0],2)+6.3*pow(10,5)*xm[0])]]))

#0.75*0.00085
sigma_nozzle75= sym(as_tensor([[-1*(-1.66*pow(10,9)*pow(xm[0],3)-1.0*pow(10,8)*pow(xm[0],2)-1.2*pow(10,4)*xm[0]), 4.52*pow(10,5)*xm[0]],[4.52*pow(10,5)*xm[0], -1*(3*pow(10,11)*pow(xm[0],3)+9.58*pow(10,8)*pow(xm[0],2)+1.8*pow(10,5)*xm[0])]]))



sigma_nozzle=sigma_nozzle100
#############################################################################################################





#######Function space definitions

#define quadrature degree
QUAD_DEG = 2

# Function space with DoFs at quadrature points:
Qelement = FiniteElement("Quadrature", mesh.ufl_cell(),
                         degree=QUAD_DEG,
                         quad_scheme="default")

Qvelement = TensorElement("Quadrature", mesh.ufl_cell(),
                         degree=QUAD_DEG,
                         quad_scheme="default")
Q = FunctionSpace(mesh, Qelement)
Qv = FunctionSpace(mesh, Qvelement)


V = VectorFunctionSpace(mesh, "CG", 1)   #for displacement
Vv = TensorFunctionSpace(mesh, "DG", 0)  #for eps_r and F_r


Y = FunctionSpace(mesh, "CG", 1)  #for temperature
Y2 = FunctionSpace(mesh, "DG", 0)   #defined for printing alpha


dx = dx(metadata={"quadrature_degree":QUAD_DEG, "quadrature_scheme": "default"})







############### Define boundary condition##############################################
def boundary(x, on_boundary):
	return on_boundary

left =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 0.0, tol=1e-7)
bottom =  CompiledSubDomain("near(x[1], side, tol) && on_boundary", side = 0.0, tol=1e-7)
right =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = x_right, tol=1e-7)
righttop = CompiledSubDomain("abs(x[0]-0.0055)<1e-7 && abs(x[1])>0.0014")
lefttop = CompiledSubDomain("abs(x[0]-0.005)<1e-7 && abs(x[1])>0.0014")

printerheight=0.00 #0.002

top = CompiledSubDomain('x[1] >= ph && abs(x[0]-xright)<1e-7', tol=1e-7, ph=printerheight, xright=x_right)
rightbottom = CompiledSubDomain('x[1] <= ph && abs(x[0]-xright)<1e-7', tol=1e-7, ph=printerheight, xright=x_right)

bottomarea = CompiledSubDomain('x[1] <= ph', tol=1e-7, ph=printerheight)
toparea = CompiledSubDomain('x[1] < ph', tol=1e-7, ph=printerheight)
toparea_prev = CompiledSubDomain('x[1] < ph', tol=1e-7, ph=printerheight)


Ttrig=Expression("T_trig",degree=1, T=T_trig) 
bcl_W = DirichletBC(Y, Ttrig, bottom)
bcs_W = [bcl_W]

c=Expression(("0.0*t","0.0*t"),degree=1,t=0)

bcc_u = DirichletBC(V.sub(0), Constant(0.0), left )
bcl_u = DirichletBC(V.sub(1), Constant(0.0), bottom )
bcr_u = DirichletBC(V, c, top )
bcs_u = [bcc_u, bcl_u, bcr_u]
bcs_du=[bcc_u, bcl_u, bcr_u]








## Define functions ########################
iota=TestFunction(Y)
theta=TrialFunction(Y)
theta_n = Function(Y)
theta_  = Function(Y)

du = TrialFunction(V)            
v  = TestFunction(V)             
u  = Function(V)                 
u_inc = Function(V)

# Function to save alpha:
alpha_ = Function(Q)
alpha_n = Function(Q)
alpha_r=Function(Y2)


uold = Function(V)
assign(uold,u)

u_int = Function(V)

epsr= Function(Vv) #Vv
epsrold=Function(Vv)

etheta= Function(Y) 
ethetaold=Function(Y)


Fr = project(Identity(2), Vv) 
iFr = project(Identity(2), Vv) 
detiFr = project(1.0, Y) 

eps33r= Function(Y) 
eps33rold=Function(Y)
F33r = project(1.0, Y) 
iF33r = project(1.0, Y) 

Fte= Function(Y)
Fcs= Function(Y) 

Cv1= project(Identity(2), Vv) 
Cv1_old= project(Identity(2), Vv) 
Cv1inv= project(Identity(2), Vv) 

Cv33_1= project(1.0, Y) 
Cv33_1_old= project(1.0, Y) 

Cv2= project(Identity(2), Vv) 
Cv2_old= project(Identity(2), Vv) 
Cv2inv= project(Identity(2), Vv) 

Cv33_2= project(1.0, Y) 
Cv33_2_old= project(1.0, Y) 

Cv3= project(Identity(2), Vv) 
Cv3_old= project(Identity(2), Vv) 
Cv3inv= project(Identity(2), Vv) 

Cv33_3= project(1.0, Y) 
Cv33_3_old= project(1.0, Y) 

E= Function(Q)
nu= Function(Q)
mu_Eq= Function(Q)
lmbda_Eq= Function(Q)

##################################################




# Define initial value
theta_init = Expression('T_0',degree=1, T_0=T_0)
theta_n.interpolate(theta_init)
theta_.interpolate(theta_init)

alpha_init = Expression('a',degree=1, a=alpha_0)
alpha_n.interpolate(alpha_init)
alpha_.interpolate(alpha_init)


u_init = Constant((0.0,  0.0))
u.interpolate(u_init)
for bc in bcs_u:
	bc.apply(u.vector())








#subdomains
printed = MeshFunction("size_t", mesh, 2)  
printed.set_all(0)
toparea.mark(printed, 1)
toparea_prev.mark(printed, 2)
dx = dx(subdomain_data=printed)

faces = MeshFunction("size_t", mesh, 1)   
faces.set_all(0)
top.mark(faces, 1)
rightbottom.mark(faces, 2)
ds = ds(subdomain_data=faces)


# Kinematics
I = Identity(2)             # Identity tensor
F = I + grad(u)             # Deformation gradient
Fer=F*inv(Fr)
C = Fer.T*Fer                   # Right Cauchy-Green tensor
F33=1+u[0]/xm[0]
Ic = tr(C) + F33*F33
J  = det(Fer)
Ce=F.T*F
Cinv=inv(Ce)
Je  = det(F)*F33
FinvT=inv(F.T)
Cstrain=1/2*(C-I)
Cstrain_r=Function(Vv)




E=E2+(E1-E2)/(1+exp(20*(alpha_-0.79)))
nu=nu2+(nu1-nu2)/(1+exp(12*(alpha_-0.65)))

mu_Eq=E/(2*(1 + nu))
lmbda_Eq=E*nu/((1 + nu)*(1 - 2*nu))








# Define solvers
solver_tol = 1.e-7

snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "cg",   #lu or gmres or cg 'preconditioner: ilu, amg, jacobi'
                                          "preconditioner": "amg",						  
                                          "maximum_iterations": 10,
                                          "report": True,
                                          "error_on_nonconvergence": False}}	

solver_u = KrylovSolver('cg', 'amg')
max_iterations=450
solver_u.parameters["maximum_iterations"] = max_iterations
solver_u.parameters["error_on_nonconvergence"] = False




# Time-stepping parameters
t = 0
tau=0
step=1
qcoeff=0.001*(10**6)
terminate=0
terminate2=0
gfcount=0
nrcount=0
rtol=1e-9
omega0=1e2
omega=omega0

minstepsize=dt/16
t_prev=0.0
tint=0.0
##################





##Steps of applying pre-stress
multiplycounter=0.1
rsmultiply=Expression("mc*c",degree=1, c=1, mc=multiplycounter )
countersteps=int(1/multiplycounter)

##############################



#Time stepping starts here
while t<=T:
	# Update current time
	t += dt
	if comm_rank==0:
		print('t= %f' %t)
	
	if terminate2==0:
		omega=omega0
	else:
		rtol=1e-7
		
		gfcount+=1
		if omega>omega0:
			omega=omega0
	
	if gfcount>6:
		gfcount=0
		omega*=10
	
	
	if t>dt and t<Ltime+t_trig:
		bct_W=bcs_W
	else:
		bct_W=[]
	
	c.t=t; 

	F33=1+u[0]/xm[0]
	Je  = det(F)*F33
	
	start_time = time.time()
	
	##############################################################################
	## Solve thermal-chemical equations for first alpha and then T.
	## Solved after a period Ltime in which the material extruded out=Ltime*Vprint
	##############################################################################
	if   t>Ltime: 
		
		local_project(alpha_n+dt*A*exp(-Er/(R*theta_n))*((1-alpha_n)**n)*(alpha_n**m)/(1+exp(Ca*(alpha_n-alpha_c))), Q, alpha_)
		alpha_.vector()[:] = np.minimum(alpha_.vector()[:],0.999)
		alpha_.vector()[:] = np.maximum(alpha_.vector()[:],alpha_0-1e-4)
		
		diff_eq = rho*Cp*theta*iota*xm[0]*dx - (rho*Cp*theta_n)*iota*xm[0]*dx + dt*kappa*inner(abs(Je)*Cinv*grad(theta), grad(iota))*xm[0]*dx  - rho*H*alpha_*iota*xm[0]*dx + rho*H*alpha_n*iota*xm[0]*dx  #+ dt*hL*(theta-(373.15-8000*xm[1]))*iota*xm[0]*ds(2)
		a_diff, L_diff = lhs(diff_eq), rhs(diff_eq)
		solve(a_diff == L_diff, theta_, bcs=bct_W,solver_parameters={"linear_solver": "cg","preconditioner": "amg"},form_compiler_parameters={"optimize": True})
	
	
	####################################################################################
	##Calculate the internal variable tensors Fr, F33r, Ftheta, Fcs, E(alpha), nu(alpha)
	####################################################################################
	
	# local_project(1-conditional(le(alpha_, 0.79), 0, 1)*cure_shrink ,Y, Fcs)
	local_project(1 ,Y, Fcs)
	local_project(1+at*(theta_-T_0) ,Y, Fte)
	local_project(epsrold+ (eigen_function(alpha_)*epsilon_elastic(u) + eigen_function(alpha_n)*epsilon_elastic(uold))*(alpha_-alpha_n)/2, Vv, epsr)
	local_project(ethetaold+ (eigen_function(alpha_)*at*(theta_-T_0) + eigen_function(alpha_n)*at*(theta_n-T_0))*(alpha_-alpha_n)/2, Y, etheta)
	local_project((Identity(2)+ epsrold+ (eigen_function(alpha_)*epsilon_elastic(u) + eigen_function(alpha_n)*epsilon_elastic(uold))*(alpha_-alpha_n)/2)/(1+etheta), Vv, Fr)
	local_project(inv(Fr),Vv,iFr)
	local_project(det(iFr)/(Fte*Fte)/(Fcs*Fcs),Y,detiFr)
	
	local_project(eps33rold+ (eigen_function(alpha_)*epsilon_33(u) + eigen_function(alpha_n)*epsilon_33(uold))*(alpha_-alpha_n)/2, Y, eps33r)
	local_project((1+eps33rold+ (eigen_function(alpha_)*epsilon_33(u) + eigen_function(alpha_n)*epsilon_33(uold))*(alpha_-alpha_n)/2)/(1+etheta), Y, F33r)
	local_project(inv(F33r),Y,iF33r)
	
	local_project(E2+(E1-E2)/(1+exp(20*(alpha_-0.79))), Q, E)
	local_project(nu2+(nu1-nu2)/(1+exp(12*(alpha_-0.65))), Q, nu)

	local_project(E/(2*(1 + nu)), Q, mu_Eq)
	local_project(E*nu/((1 + nu)*(1 - 2*nu)), Q, lmbda_Eq)
	
	if comm_rank==0:
		print("--- %s seconds ---" % (time.time() - start_time))
	
	
	
	
	###############################################################################################################
	##Solve for the displacement field
	##Extrude the material according to Vflow and move the printer according to Vprint, when time t_add is reached
	##This process is done in two steps. 
	##First, we remove the boundary condition on a region corresponding to Vflow in the reference configuration.
	##This means Vflow*tadd material has been extruded out, and the printer has moved up by Vflow*tadd also. The
	## pre-stress is applied in steps in this newly extruded material.
	## Second, we move the printerhead into position by applying a displacement of (Vprint-Vflow)*(tadd).
	###############################################################################################################
	
	start_time = time.time()
	
	
	if step % addsteps==0:  #==1
	
		#top is the boundary of the region inside the nozzle. This boundary moves along with the printerhead coordinates prescribed through Vprint
		#toparea_prev represents the region that has already been extruded out before t
		#toparea represents the region extruded out at time t. In this region, the pre-stress is applied in steps for extra stability
		
		top = CompiledSubDomain('x[1] >= ph+Vflow*(t-t0)-1e-6  && abs(x[0]-xright)<1e-7', tol=1e-7, t=t, t0=tint, ph=printerheight, xright=x_right, Vflow=Vflow) 
		toparea_prev= CompiledSubDomain('x[1] < ph+Vflow*(tprev-t0)', tol=1e-7, tprev=t_prev, t0=tint, ph=printerheight, dt=dt, Vflow=Vflow)
		toparea = CompiledSubDomain('x[1] < ph+Vflow*(t-t0) && x[1] >= ph+Vflow*(tprev-t0)', tol=1e-7, t=t, tprev=t_prev, t0=tint, ph=printerheight, dt=dt, Vflow=Vflow)
		c2=Expression(("0.0","(Vprint-Vflow)*(t-t0)"),degree=1,t=t, t0=tint, tprev=t_prev, Vflow=Vflow, Vprint=Vprint)
		bcr_u = DirichletBC(V, c2, top )
		bcs_u = [bcc_u, bcl_u, bcr_u]
		
		
		bcr_du = DirichletBC(V, c, top )
		bcs_du=[bcc_u, bcl_u, bcr_du]
		
		printed = MeshFunction("size_t", mesh, 2)  
		printed.set_all(0)
		toparea.mark(printed, 1)
		toparea_prev.mark(printed, 2)
		dx = dx(subdomain_data=printed)
		
		rightbottom = CompiledSubDomain('x[1] < ph+Vflow*(t-t0)-1e-6 && abs(x[0]-xright)<1e-7', tol=1e-7, t=t, t0=tint, ph=printerheight, xright=x_right, Vflow=Vflow)
		rightbottom.mark(faces, 2) 
		ds = ds(subdomain_data=faces)
		
		
		#Applying pre-stress in steps
		counter=1
		while counter<=countersteps:
			
			psi_eq=equilbriumbranch(u,Fte,Fcs,iFr,iF33r,detiFr)
			psi_neq1=nonequilbriumbranch(u,Fte,Fcs,Cv1,Cv1inv,Cv33_1,mu_Neq1,lmbda_Neq1)
			psi_neq2=nonequilbriumbranch(u,Fte,Fcs,Cv2,Cv2inv,Cv33_2,mu_Neq2,lmbda_Neq2)
			psi_neq3=nonequilbriumbranch(u,Fte,Fcs,Cv3,Cv3inv,Cv33_3,mu_Neq3,lmbda_Neq3)
			
			rsmultiply.c=counter
			Pi = psi_eq*xm[0]*dx + (psi_neq1+psi_neq2+psi_neq3)*xm[0]*dx   - dot(B, u)*xm[0]*dx 
			R_u = derivative(Pi, u, v) - inner(rsmultiply*Je*sigma_nozzle*FinvT,grad(v))*xm[0]*dx(1) - inner(Je*sigma_nozzle*FinvT,grad(v))*xm[0]*dx(2) 
			Jac_u = derivative(R_u, u, du)
			Jac3=inner(grad(du),grad(v))*xm[0]*dx
			
			terminate=0
			u,terminate = implicitgradientflow(R_u, Jac_u, Jac3, bcs_du, bcs_du, terminate, terminate2, omega  )
			
			if terminate==1:
				break
			
			counter += 1
		
		if terminate==0:
			
			#Moving the printerhead into position in 10 steps
			bccounter=1
			while bccounter<=10:
				c2=Expression(("0.0","(Vprint-Vflow)*(tprev-t0)+0.1*c*(Vprint-Vflow)*(t-tprev)"),degree=1,t=t, t0=tint, tprev=t_prev, Vflow=Vflow, Vprint=Vprint, c=bccounter)
				bcr_u = DirichletBC(V, c2, top )
				bcs_u = [bcc_u, bcl_u, bcr_u]
				
				c4=Expression(("0.0","0.1*(Vprint-Vflow)*(t-tprev)"),degree=1,t=t, t0=tint, tprev=t_prev, Vflow=Vflow, Vprint=Vprint)
				bcr_du0 = DirichletBC(V, c4, top )
				bcs_du0=[bcc_u, bcl_u, bcr_du0]
				
				psi_eq=equilbriumbranch(u,Fte,Fcs,iFr,iF33r,detiFr)
				psi_neq1=nonequilbriumbranch(u,Fte,Fcs,Cv1,Cv1inv,Cv33_1,mu_Neq1,lmbda_Neq1)
				psi_neq2=nonequilbriumbranch(u,Fte,Fcs,Cv2,Cv2inv,Cv33_2,mu_Neq2,lmbda_Neq2)
				psi_neq3=nonequilbriumbranch(u,Fte,Fcs,Cv3,Cv3inv,Cv33_3,mu_Neq3,lmbda_Neq3)
				
				Pi = psi_eq*xm[0]*dx + (psi_neq1+psi_neq2+psi_neq3)*xm[0]*dx  - dot(B, u)*xm[0]*dx 
				R_u = derivative(Pi, u, v) - inner(Je*sigma_nozzle*FinvT,grad(v))*xm[0]*dx(1) - inner(Je*sigma_nozzle*FinvT,grad(v))*xm[0]*dx(2)#+ (qcoeff/h_avg)*dot(jump(u),jump(v))*dS
				Jac_u = derivative(R_u, u, du)
				Jac3=inner(grad(du),grad(v))*xm[0]*dx
				
				u,terminate = implicitgradientflow(R_u, Jac_u, Jac3, bcs_du0, bcs_du, terminate, terminate2, omega  )
				
				if terminate==1:
					break
				
				bccounter += 1
		
		if terminate==0:
			t_prev= t
		
	else:
		#At all times when t%tadd ~=0, no new material is added but the front progresses. So the following is solved
		psi_eq=equilbriumbranch(u,Fte,Fcs,iFr,iF33r,detiFr)
		psi_neq1=nonequilbriumbranch(u,Fte,Fcs,Cv1,Cv1inv,Cv33_1,mu_Neq1,lmbda_Neq1)
		psi_neq2=nonequilbriumbranch(u,Fte,Fcs,Cv2,Cv2inv,Cv33_2,mu_Neq2,lmbda_Neq2)
		psi_neq3=nonequilbriumbranch(u,Fte,Fcs,Cv3,Cv3inv,Cv33_3,mu_Neq3,lmbda_Neq3)
		
		Pi = psi_eq*xm[0]*dx + (psi_neq1+psi_neq2+psi_neq3)*xm[0]*dx  - dot(B, u)*xm[0]*dx 
		R_u = derivative(Pi, u, v) - inner(Je*sigma_nozzle*FinvT,grad(v))*xm[0]*dx(1) - inner(Je*sigma_nozzle*FinvT,grad(v))*xm[0]*dx(2) #+ (qcoeff/h_avg)*dot(jump(u),jump(v))*dS
		Jac_u = derivative(R_u, u, du)
		Jac3=inner(grad(du),grad(v))*xm[0]*dx
		
		terminate=0
		u,terminate = implicitgradientflow(R_u, Jac_u, Jac3, bcs_du, bcs_du, terminate, terminate2, omega) 

	
	if comm_rank==0:
		print("--- %s seconds ---" % (time.time() - start_time))
	
	
	##################################################################################
	## Solve the evolution equation for internal variable Cv with a fifth order scheme
	##################################################################################
	
	start_time = time.time()
	
	kstep1, k33step1=k_terms(dt, u, uold, Cv1_old, Cv33_1_old, Fte, Fcs, mu_Neq1, vis1)
	kstep2, k33step2=k_terms(dt, u, uold, Cv2_old, Cv33_2_old, Fte, Fcs, mu_Neq2, vis2)
	kstep3, k33step3=k_terms(dt, u, uold, Cv3_old, Cv33_3_old, Fte, Fcs, mu_Neq3, vis3)
	
	local_project(Cv1_old + kstep1, Vv, Cv1)
	local_project(inv(Cv1), Vv, Cv1inv)
	local_project(Cv33_1_old + k33step1, Y, Cv33_1)
	
	local_project(Cv2_old + kstep2, Vv, Cv2)
	local_project(inv(Cv2), Vv, Cv2inv)
	local_project(Cv33_2_old + k33step2, Y, Cv33_2)
	
	local_project(Cv3_old + kstep3, Vv, Cv3)
	local_project(inv(Cv3), Vv, Cv3inv)
	local_project(Cv33_3_old + k33step3, Y, Cv33_3)
	

	if comm_rank==0:
		print("--- %s seconds ---" % (time.time() - start_time))
	
	
	
	if terminate==0:
		# Save to file 
		if step % printsteps==0 or step==1:
			local_project(alpha_, Y2, alpha_r)
			file_results = XDMFFile( "/fenics/visco-printhead-pillar-Axi/pillarAxiM3vis150_flexlayer25bc_L1_Vp85Vf128_" + str(step) + ".xdmf" )
			file_results.parameters["flush_output"] = True
			file_results.parameters["functions_share_mesh"] = True
			theta_.rename("theta", "temperature")
			alpha_r.rename("alpha", "degree of cure")
			u.rename("u", "displacement field")
			file_results.write(u,t)
			file_results.write(theta_,t)
			file_results.write(alpha_r,t)
			

		assign(uold,u)
		assign(epsrold,epsr)
		assign(eps33rold,eps33r)
		assign(ethetaold,etheta)
		theta_n.assign(theta_)
		alpha_n.assign(alpha_)
		
		assign(Cv1_old,Cv1)
		assign(Cv33_1_old,Cv33_1)
		assign(Cv2_old,Cv2)
		assign(Cv33_2_old,Cv33_2)
		assign(Cv3_old,Cv3)
		assign(Cv33_3_old,Cv33_3)
		step+=1
		tau+=dt
		
	else:
		t -= dt
		assign(u,uold)
		assign(epsr,epsrold)
		assign(eps33r,eps33rold)
		assign(etheta,ethetaold)
		theta_.assign(theta_n)
		alpha_.assign(alpha_n)
		
		assign(Cv1, Cv1_old)
		assign(Cv33_1, Cv33_1_old)
		assign(Cv2, Cv2_old)
		assign(Cv33_2, Cv33_2_old)
		assign(Cv3, Cv3_old)
		assign(Cv33_3, Cv33_3_old)
		
		if gfcount>0:
			gfcount-=1
		omega*=0.1
		terminate2=1
		