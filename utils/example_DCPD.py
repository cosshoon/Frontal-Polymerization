# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
#
# An example of running the FP solver for DCPD

import os
from solver_FP import FP_solver
from kinetics_library import Prout_Tompkins_Diffusion
from postProcess_FP import computeFrontSpeed, frontAnimation, saveResults, loadResults


# Switch to current directory 
try:
    os.chdir(os.path.dirname(__file__))
except:
    print("Cannot switch directories")

# Thermo-chemical properties
kr =  0.15      # thermal conductivity, W/m-K
rhor = 980.0    # density, kg/m^3
Cpr = 1600.0    # specific heat, J/kg-K

# Nature values
Hr =  350000    # enthalpy, J/kg
A_ = 8.55e15    # pre-exponential constant, 1/s
Er = 110750.0   # activation energy J/mol
n = 1.72        # cure kinetics parameter
m = 0.77        # cure kinetics parameter
Ca = 14.48      # cure kinetics parameter
alpha_c = 0.41  # cure kinetics parameter

# Create an object for the cure kinetics model
ptd_model = Prout_Tompkins_Diffusion(n, m, Ca, alpha_c)

# Initial conditions for temperature and degree of cure
T0 = 20         # deg C
alpha0 = 0.01

# Length of domain
L = 0.01

# Initialize the solver with the parameters
solver = FP_solver(kr, rhor, Cpr, Hr, A_, Er, ptd_model, T0, alpha0, L=L, h_x=2e-5)

# Run the solver until 10 s
t_end = 10
x_data, t_data, T_data, alpha_data = solver.solve(t_end, outputFreq_t=10, outputFreq_x=1)

# Post-process - compute the front speed, and save an animation of the temperature and alpha fronts
V_front = computeFrontSpeed(x_data, t_data, T_data, alpha_data, save_dir="out", save_suffix="DCPD", fid=0)
frontAnimation(x_data, t_data, T_data, alpha_data, save_dir="out", save_suffix="DCPD", anim_frames=100, fid=1)

# Pack the results for future use
saveData = {}
saveData['x_data'] = x_data
saveData['t_data'] = t_data
saveData['alpha_data'] = alpha_data
saveData['Temp_data'] = T_data

saveResults(saveData,"out","data_DCPD") # save to .mat file
loadedResults = loadResults("out","data_DCPD") # load from .mat file