# User story 3 : Change flight mode for test from Mission Planner
- As a GC Operator, I want to be able to change the type of the flight test from Mission Planner, so I can have a streamlined interface to set the flight test type to the flying UAS
 
## AC 3.1 Good input -> Good outcome (GC Operator puts in correct values and drone executes test)
- Given GCO sets correct param [SID_TYPE] between 1-6, when GCO sets param THEN the codebase will read the parameter THEN the code base will create the signal THEN send the control trajectory to the UAS AND then the UAS will conduct manuever specified from SID_TYPE
 
## AC 3.2 Bad input -> Good outcome (GC Operator puts in correct values and drone executes test)
- Given GCO sets wrong param [SID_TYPE] beyond 6 or is not numeric, when GCO sets param THEN the codebase will read the parameter THEN the code base will reject the signal AND the UAS will continue doing its current manuever
