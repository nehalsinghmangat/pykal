Cookbook & Recipes
==================

.. hlist::
   :columns: 2

   * :doc:`installation`
   * :doc:`quickstart`
   * :doc:`theory`
   * :ref:`cookbook_create_model`
   * :ref:`cookbook_tune_noise`
   * :ref:`cookbook_plot_metrics`

.. _cookbook_create_model:

Creating a Custom Model
-----------------------

Demonstrates how to supply your own nonlinear `f` and `h`.

.. code-block:: python

   from pykal import EKF
   import numpy as np

   def my_dynamics(x, u=None, t=None):
       return np.sin(x)

   ekf = EKF(my_dynamics, my_dynamics, Q=np.eye(1)*1e-3, R=np.eye(1)*1e-2)
