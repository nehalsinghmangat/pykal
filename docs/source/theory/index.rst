
Theory & Background
===================

This section provides a concise yet rigorous overview of the theoretical concepts behind the estimators in **pykal**, starting from inverse problems and least-squares estimation, and culminating in the Kalman Filter.

.. toctree::

   :maxdepth: 2
   :caption: Topics in Estimation

   inverse_problems_and_least_squares
   motivating_example_sensor_calibration


.. admonition:: Observability Summary

   A state or parameter is observable if it can be uniquely inferred from data.

   - **Batch setting:** Observability ⇔ full column rank of design matrix :math:`A`
   - **Sequential setting:** Observability ⇔ persistent excitation of :math:`a_k`
   - **Quantified by:** Invertibility of the observability Grammian :math:`W = A^T A`
