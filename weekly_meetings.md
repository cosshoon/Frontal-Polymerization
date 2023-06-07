## Details of weekly meetings

2/7/23

- [ ] Task 1: Compute the energy contributions due to polymerization (E<sub>p</sub>) and internal energy (E<sub>th</sub>), as well as heat lost over surface from heat convection and through inlet.
  - Refer to the method used to compute heat flux in original code: 
  https://github.com/mzakoworotny/FP_Printing_Modeling/blob/bec81ceb440c3684ec70f749c35b414ec20e5e46/printing/composites/roller_printer/composite_roller_steady_nonDim.py#L510
  Note that dx is the measure used for area integrals, and ds the outer boundary ds is partitioned into : 
  https://github.com/mzakoworotny/FP_Printing_Modeling/blob/bec81ceb440c3684ec70f749c35b414ec20e5e46/printing/composites/roller_printer/composite_roller_steady_nonDim.py#L231-L236
  Refer to Slide 17 for the non-dimensional versions of the relevant terms: https://github.com/mzakoworotny/FP_Printing_Modeling/blob/main/printing/composites/roller_printer/composite_printing_model.pptx for
  - Try this for 2 cases - one where extrusion speed is lower than the natural front speed (0.5 mm/s) and another that is greater than the front speed (3 mm/s)
  
- [ ] Task 2: Implement the transient version of the composite printing model, for the non-dimensional system of equations. For reference, use the existing non-dimensional steady-state code, and the dimensional transient code. The only change to the weak form should be adding the time derivative terms to the non-dimensional weak from in the steady state code.

  Weak form from transient model (dimensional):
  https://github.com/mzakoworotny/FP_Printing_Modeling/blob/8381d2c328946f394d576915c431bfd76c566f6c/printing/composites/roller_printer/transient/composite_roller_transient_dimensional.py#L274-L283
  
  Weak form from steady-state model (non-dimensional):
  https://github.com/mzakoworotny/FP_Printing_Modeling/blob/278c57579a80a56c74067df23c05ae3560620499/printing/composites/roller_printer/composite_roller_steady_nonDim.py#L319-L333
  
  For the solver, refer to the method for time-stepping I used in the old transient code:
  https://github.com/mzakoworotny/FP_Printing_Modeling/blob/8381d2c328946f394d576915c431bfd76c566f6c/printing/composites/roller_printer/transient/composite_roller_transient_dimensional.py#L318-L343
  
  This code will be important for investigating non-steady heating conditions, such as cases where the heater is turned off to allow for front to propagate naturally.

3/2/23

- Discussed some recent results on composite printing using an extended domain for the tow upstream and downstream of the roller. Found that the front was able to advance ahead of the rollers when Vr < Vfront, if the roller temperature is sufficiently high.
- Discussed preliminary results on the "energy balance" at the roller-tow contact surface. In the current model, there is some energy input required to maintain the surface at a certain temperature Tr (since it is higher than T0), and there is some energy released due to the exothermic FP reaction. At low Vr, we find that more energy is released by FP than is required to maintain Tr, indicating we can stop externally heating the tow. At high Vr, the energy released by FP is too low, so the external heating must be maintained.
- [ ] However, this model for energy is not representative of the actual printing setup, since the tow should be able to heat up due to FP. To better simulate this process, we will use a new model that explicitly includes the roller and the thermal contact resistance with the tow. This will introduce a discontinuity of temperature that should allow the tow to heat up above Tr. The goal is to find that at low Vr, no energy is required to be input to the system since FP causes the front to propagate naturally.
