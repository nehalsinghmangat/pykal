=====================
Citation Example Page
=====================

This page demonstrates how to cite papers from the bibliography.

.. note::
   This is an example page. You can delete it or use it as a template.

Introduction
============

The Kalman filter :cite:`kalman1960new` is a fundamental tool in state estimation.
It provides optimal estimates for linear systems under Gaussian noise assumptions.

For nonlinear systems, the Unscented Kalman Filter :cite:`julier1997new` offers
an alternative approach that doesn't require Jacobian calculations.

Implementation in pykal
=======================

Our implementation of the Kalman filter can be found in ``pykal.utilities.estimators.kf``.
The implementation follows the formulation presented in :cite:`kalman1960new`.

See Also
========

For a complete list of papers and algorithms, visit the :doc:`bibliography` page,
where you can filter by algorithm type, robot platform, and implementation status.

References
==========

.. bibliography::
   :filter: cited
