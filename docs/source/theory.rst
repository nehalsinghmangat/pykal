
Theory & Background
===================

This section provides a concise overview of the filter variants in **pykal**.

Standard Kalman Filter (KF)
---------------------------
- Optimal for linear, Gaussian systems  
- Prediction:  
  \[ x_{k+1} = F\,x_k + w_k,\quad w_k\sim\mathcal{N}(0,Q)\]  
- Update:  
  \[ y_k = H\,x_k + v_k,\quad v_k\sim\mathcal{N}(0,R)\]  

Extended Kalman Filter (EKF)
-----------------------------
- Linearises nonlinear models about the current estimate  
- Uses Jacobians \(F\) and \(H\) in place of constants

Schmidt-Kalman Filter (SKF)
----------------------------
- Freezes a subset of “nuisance” states during update  
- Useful when part of the state is known to be unobservable

Partial-Update SKF (PSKF)
-------------------------
- Generalises SKF with per-state weights \(\beta_i\in[0,1]\)  
- Allows soft updates (Brink 2017)

Observability-Informed PSKF (OPSKF)
-----------------------------------
- Dynamically chooses \(\beta\) based on observability metrics  
  – “nullspace” test or “stochastic” covariance proxy (Ramos et al. 2021)