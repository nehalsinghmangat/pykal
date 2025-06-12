Quickstart
==========

A minimal end-to-end EKF example on a 1D constant-velocity model.

1. Define your dynamics and measurement models:

   .. code-block:: python

      import numpy as np
      from pykal import EKF, SystemType

      # x = [position, velocity]
      def f(x, u=None, t=None):
          # constant velocity
          return np.array([x[1], 0.0])

      def h(x, u=None, t=None):
          # measure position only
          return np.array([x[0]])

2. Set up noise covariances:

   .. code-block:: python

      Q = np.diag([1e-4, 1e-4])   # process noise
      R = np.array([[1e-2]])      # measurement noise

3. Instantiate and run:

   .. code-block:: python

      ekf = EKF(f, h, Q, R,
                system_type=SystemType.CONTINUOUS_TIME_INVARIANT,
                dt=0.1)

      # initial state & covariance
      x0 = np.array([0.0, 1.0])
      P0 = np.eye(2)

      # simulate noisy measurements
      true_positions = np.linspace(0, 10, 101)
      zs = true_positions + np.random.normal(scale=0.1, size=101)
      x_est, P_est = ekf.run(x0, P0, zs)

4. Plot results:

   .. code-block:: python

      ekf.plot_kf_predictions(x_est, P_est)