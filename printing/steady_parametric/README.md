Script for transient FP printing, under constant process parameters (Vgel, Vprint, ambient temperature), to observe steady state behavior of system
Model accounts for solidification of gel by increase in viscosity, thermal expansion/cure shrinkage, gravity

Important considerations when running:

Make sure you have meshio and gmsh Python modules installed

Under parameters, first set the desired v_gel_max, v_print_max, T_amb and h_conv.
For now, use R = 0.00077 m.
Need to set L_swell (length of the domain outside the nozzle) such that the front is contained in the computational domain.
Then, need to set L_front_refine (distance outside the nozzle at which the mesh becomes refined, which is needed for accurate solution of thermo-chemical equations). Using an estimate of the steady state location of the front, Lb, set L_front_refine to be less than this estimate.

Run on 8 - 12 cores.

If script gets hung up when generating mesh (happens occassionally with gmsh when generating mesh in parallel), cancel simulation (Ctrl-C) and run again.
