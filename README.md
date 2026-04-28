# BMI: Causal Decoding of Reach Trajectories from Motor Cortex
Group 3 - Monkey Brain


--------------------------------------------------------------------------------
Project Overview
--------------------------------------------------------------------------------
Our team implemented a two-stage decoding pipeline:

1) Direction Classification: We utilised a two-stage Linear Discriminant Analysis (LDA) classifier, which outperformed K-Nearest Neighbours and Support Vector Machines.

2) Feature Extraction: Multi-timescale lag features were constructed from spike counts and reduced via Principal Component Analysis (PCA)

3) Trajectory Regression: We employed segmented ridge regression to map neural features to hand displacement and velocity across eight distinct reach phases.

4) State-Space Modelling: A direction-specific Kalman Filter enforces temporal smoothness

5) Endpoint Correction: A "target-pull" mechanism applies a soft prior towards the predicted target's final position after 400 ms to reduce late-trial drift.

--------------------------------------------------------------------------------
Run
--------------------------------------------------------------------------------
To evaluate the model performance, ensure you are in the root directory in MATLAB 
and execute:

RMSE = testFunction_for_students_MTb('MonkeyBrain')

--------------------------------------------------------------------------------
Results
--------------------------------------------------------------------------------
Accuracy: 98.8% direction-classification accuracy.  
Error: ~8.99 mm RMSE

--------------------------------------------------------------------------------
Members
--------------------------------------------------------------------------------
- Ariel Ang
- Balvinder Kaur Dhillon
- Jorge Gomez Aguilar
- Dineth Ilapperuma
- Morgan Helene

For the module BIOE70011 Brain Machine Interfaces @ Imperial College London (2025-2026)
